import copy
import time
import heapq
import numpy as np
import torch
import torch.utils.data as tud
from heapq import nlargest
from typing import List, Tuple, Union, Optional
from reinvent_chemistry import Conversions
from pepinvent.reinforcement.utils import softmax, grpo_normalize, print_, plot_argmax_indices, animate_router_distribution, save_probability_histogram, mask_chuckles, fill_chuckles, get_mask_smiles_visualization
from pepinvent.scoring_function.scoring_components.custom.utils import clean_memory
from pepinvent.reinforcement.router import Router
from pepinvent.reinforcement.chemistry import Chemistry
from pepinvent.reinforcement.configuration.reinforcement_learning_configuration import ReinforcementLearningConfiguration
from pepinvent.reinforcement.diversity_filters.diversity_filter import DiversityFilter
from pepinvent.reinforcement.dto.likelihood_dto import LikelihoodDTO
from pepinvent.reinforcement.dto.output_dto import OutputDTO
from pepinvent.reinforcement.dto.scoring_input_dto import ScoringInputDTO
from pepinvent.reinvent_logging.local_reinforcement_logger import LocalReinforcementLogger
from pepinvent.scoring_function.score_summary import FinalSummary
from pepinvent.scoring_function.scoring_function_factory import ScoringFunctionFactory
from reinvent_models.model_factory.dto.sampled_sequence_dto import SampledSequencesDTO
from reinvent_models.model_factory.mol2mol_adapter import Mol2MolAdapter
from reinvent_models.mol2mol.dataset.dataset import Dataset


class LearningScenario:
    def __init__(self, agent: Mol2MolAdapter, prior: Mol2MolAdapter, config: ReinforcementLearningConfiguration):
        self._prior = prior
        self._config = config
        self._logger = LocalReinforcementLogger(self._config.logging)
        self._diversity_filter = DiversityFilter(config.diversity_filter)
        self._chemistry = Chemistry()
        self._conversions = Conversions()
        self._scoring_function = ScoringFunctionFactory(config.scoring_function).create_scoring_function()
        self._lc = self._config.learning_configuration
        self._input_monomers = self._config.input_sequence.split('|')
        self._num_agents = len(self._input_monomers)
        self._topk_list = []
        self._total_steps = (
              (self._lc.warmup_steps * self._num_agents) 
            + (self._lc.routing_steps)
            + (self._lc.evolving_steps * self._lc.num_mutations)
        )
        
        (
            self._agents,
            self._agents_baseline,
            self._agents_optimizer,
            self._agents_warmup_score,
        ) = self._spawn_agents(agent=agent, num_agents=self._num_agents)

        (
            self._router, 
            self._router_baseline, 
            self._router_anneal_steps,
            self._router_optimizer, 
            self._router_accumulated_p, 
            self._router_ps,
            self._router_img, 
            self._router_argmax_idx
        ) = self._spawn_router(num_agents=self._num_agents)

    def _sample(self, agent_idx: int = None, input: Union[str, List[str]] = None, agent: Mol2MolAdapter = None, period: str = None) -> List[SampledSequencesDTO]:
        if not agent: 
            if self._lc.evolve_mode == 'single' and period == 'evolve': 
                agent_idx = self._lc.evolve_positions[0]
            agent = self._agents[agent_idx]
        if isinstance(input, str):
            input = self._lc.batch_size * [input]

        dataset = Dataset(input, vocabulary=agent.vocabulary, tokenizer=agent.tokenizer)
        data_loader = tud.DataLoader(
            dataset, self._lc.batch_size, shuffle=False, collate_fn=Dataset.collate_fn
        )
        
        results: List[SampledSequencesDTO] = []

        with torch.no_grad():
            for batch in data_loader:
                src, src_mask = batch
                batch_dtos = agent.sample(src, src_mask)
                results.extend(batch_dtos if isinstance(batch_dtos, list) else [batch_dtos])
                clean_memory(src, src_mask)

        return results

    def _score(self, agent_idx: int, input: Union[str, List[str]], sampled_sequences_dto: List[SampledSequencesDTO], step: int, period: str) -> Tuple[FinalSummary, List[SampledSequencesDTO]]:
        outputs = [dto.output for dto in sampled_sequences_dto]
        if isinstance(input, str):
            chuckles = [fill_chuckles(input, output) for output in outputs]
            peptides = [self._chemistry.fill_source_peptide(input, output) for output in outputs]
            scoring_input = ScoringInputDTO(peptides=peptides, peptide_input=input,
                                        peptide_outputs=outputs, chuckles=chuckles)
        else: 
            chuckles = [fill_chuckles(input_sequence, output) for input_sequence, output in zip(input, outputs)]
            peptides = [self._chemistry.fill_source_peptide(input_sequence, output) for input_sequence, output in zip(input, outputs)]
            scoring_input = ScoringInputDTO(peptides=peptides, peptide_input='placeholder',
                                        peptide_outputs=outputs, chuckles=chuckles)
        # uniqueness check
        sorted_indices = self._get_indices_of_unique_smiles(scoring_input.peptides)
        scoring_input_unique, sampled_sequences_dto_unique = self._keep_unique(scoring_input, sorted_indices, sampled_sequences_dto)
        print(f" unique molecules:{len(scoring_input_unique.peptides)}\n")
        final_summary = self._scoring_function.calculate_score(scoring_input_unique)
        final_summary.monomers = [dto.output for dto in sampled_sequences_dto_unique]
        final_summary.total_score = self._diversity_filter.update_score(final_summary, sampled_sequences_dto_unique, step, period, mask_pos=agent_idx)
        return final_summary, sampled_sequences_dto_unique, sorted_indices

    def _update(self, agent_idx: int, final_summary: FinalSummary, sampled_sequences_dto: List[SampledSequencesDTO], period: str) -> LikelihoodDTO:
        if self._lc.evolve_mode == 'single' and period == 'evolve':
            agent_idx = self._lc.evolve_positions[0]
        agent = self._agents[agent_idx]

        # Process in smaller chunks
        chunk_size = 8
        agent_likelihoods = []
        prior_likelihoods = []
        
        for i in range(0, len(sampled_sequences_dto), chunk_size):
            chunk = sampled_sequences_dto[i:i+chunk_size]
            
            # Calculate likelihoods for chunk
            agent_likelihood_chunk = -agent.likelihood_smiles(chunk).likelihood
            
            with torch.no_grad():  # Prior doesn't need gradients
                prior_likelihood_chunk = -self._prior.likelihood_smiles(chunk).likelihood
            
            agent_likelihoods.append(agent_likelihood_chunk)
            prior_likelihoods.append(prior_likelihood_chunk)
        
        # Concatenate chunks
        agent_likelihood = torch.cat(agent_likelihoods)
        prior_likelihood = torch.cat(prior_likelihoods)
        
        distance_penalty = self._get_distance_to_prior(prior_likelihood, self._lc.distance_threshold)

        score_tensor = final_summary.total_score * distance_penalty
        score_tensor = torch.from_numpy(score_tensor)
        score_tensor = self.to_tensor(score_tensor)

        if self._lc.agent_baseline_weight != 0 : 
            self._agents_baseline[agent_idx] = (
                self._lc.agent_baseline_weight * self._agents_baseline[agent_idx]
                + (1 - self._lc.agent_baseline_weight)
                * (score_tensor.sum().detach() / self._lc.batch_size)
            )        

        if period == 'evolve':
            score_tensor = torch.clamp(score_tensor, -1, 1)
        else: 
            # score_tensor = score_tensor - self._agents_baseline[agent_idx]
            score_tensor = score_tensor - 0 

        augmented_tensor: torch.Tensor = prior_likelihood + self._lc.score_multiplier * score_tensor
        likelihoods = LikelihoodDTO(prior_likelihood=prior_likelihood, 
                                    agent_likelihood=agent_likelihood,
                                    augmented_likelihood=augmented_tensor)

        loss = torch.pow((augmented_tensor - agent_likelihood), 2)
        loss = loss.mean()

        # Clear gradients before backward
        self._agents_optimizer[agent_idx].zero_grad()
        loss.backward()
        self._agents_optimizer[agent_idx].step()
        
        # Detach tensors to prevent gradient accumulation
        likelihoods.prior_likelihood = likelihoods.prior_likelihood.detach()
        likelihoods.agent_likelihood = likelihoods.agent_likelihood.detach()
        likelihoods.augmented_likelihood = likelihoods.augmented_likelihood.detach()
        
        # Clear GPU cache
        clean_memory(agent_likelihoods, prior_likelihoods)
        
        return likelihoods

    def _update_router(self, final_summary: FinalSummary, mask_idx: int, step: int = None) -> np.ndarray :
        r = float(final_summary.total_score.sum() / self._lc.batch_size)
        adv = r - self._router_baseline
        self._router_baseline = self._lc.router_baseline_weight * self._router_baseline + (1 - self._lc.router_baseline_weight) * r
        logp = self._router.log_prob(mask_idx)

        # Entropy annealing
        if step >= self._router_anneal_steps : 
            beta = float(self._lc.router_entropy_end) 
        else : 
            t = float(step) / float(self._router_anneal_steps) 
            beta = float(self._lc.router_entropy_start) + t * (float(self._lc.router_entropy_end) - float(self._lc.router_entropy_start))

        if self._router_anneal_steps == 0 : beta = 0

        loss_router = -(adv * logp) - (beta * self._router.entropy()) 
        self._router_optimizer.zero_grad(); loss_router.backward(); self._router_optimizer.step()

        with torch.no_grad() : 
            return self._router.forward().cpu().numpy()

    def _log(self, agent_idx, start_time, step, final_summary, likelihoods, report: List[OutputDTO]):
        self._logger.timestep_report(agent_idx, start_time, self._total_steps, step, final_summary,
                                     likelihoods.agent_likelihood,
                                     likelihoods.prior_likelihood, likelihoods.augmented_likelihood, report)

    def _run(self, agent_idx: int, period: str) : 
        input_sequence = mask_chuckles(self._config.input_sequence, agent_idx)
        print(f'input_sequence: {input_sequence}')
        agent_idx = 0
        dtos = self._sample(agent_idx, input_sequence, period=period) 
        final_summary, dtos, _ = self._score(agent_idx, input_sequence, dtos, self._global_step, period)
        
        likelihoods = self._update(agent_idx, final_summary, dtos, period)

        report = self._create_report(agent_idx, dtos, final_summary)
        self._log(agent_idx, self._start_time, self._global_step, final_summary, likelihoods, report)

        clean_memory(dtos, likelihoods, report)

        self._global_step += 1

        return final_summary
    
    def _init_evolve(self, period='init_evolve'): 
        input_sequence = mask_chuckles(self._config.input_sequence, self._lc.evolve_positions) 
        dtos = self._sample(input=input_sequence, agent=self._prior, period=period)
        final_summary, dtos, _ = self._score(0, input_sequence, dtos, self._global_step, period)
        self._update_topk_list(final_summary, k=self._lc.K)

    def _evolve(self, input_sequences: List[str], agent_idx: int, top_agents_idx: List[int], period: str): 
        K, G, num_original = self._lc.K, self._lc.G, self._lc.num_original

        original_sequence = mask_chuckles(self._config.input_sequence, top_agents_idx)
        input_sequences = [x for x in input_sequences for _ in range(G)]
        if self._lc.agent_mask_view == 'neighbor_view':
            input_sequences = [mask_chuckles(i, [x for x in top_agents_idx if x != agent_idx]) for i in input_sequences] + ([original_sequence]*num_original)
        elif self._lc.agent_mask_view == 'self_view':
            input_sequences = [mask_chuckles(i, agent_idx) for i in input_sequences] + ([original_sequence]*num_original)
        else: 
            raise ValueError(f"Unknown agent_mask_view: {self._lc.agent_mask_view}")

        dtos = self._sample(agent_idx, input_sequences, period)

        final_summary, dtos, sorted_indices = self._score(agent_idx, input_sequences, dtos, self._global_step, period)
        
        final_summary_grpo = copy.deepcopy(final_summary)
        group_ids = [i for i in range(K) for _ in range(G)]
        group_ids_filtered = [group_ids[i] for i in sorted_indices]
        singleton_idx = [idx for idx, id in enumerate(group_ids_filtered) if group_ids_filtered.count(id) == 1]

        # Remove elements with indices in singleton_idx
        final_summary_grpo.total_score = np.delete(final_summary_grpo.total_score, singleton_idx)
        dtos_grpo = [dto for idx, dto in enumerate(dtos) if idx not in singleton_idx]

        if len(dtos_grpo) == 0: 
            self._global_step += 1 
            return 
            
        group_ids_filtered = [x for i, x in enumerate(group_ids_filtered) if i not in singleton_idx]
        final_summary_grpo.total_score = grpo_normalize(final_summary_grpo.total_score, group_ids_filtered)

        likelihoods = self._update(agent_idx, final_summary_grpo, dtos_grpo, period) 
        report = self._create_report(agent_idx, dtos, final_summary)
        self._log(agent_idx, self._start_time, self._global_step, final_summary, likelihoods, report)
        self._update_topk_list(final_summary, k=K)
        self._global_step += 1

    def execute(self):
        self._global_step = 0
        self._start_time = time.time()
        
        #####################################
        ########### WARMUP PERIOD ###########
        #####################################
        for warmup_step in range(self._lc.warmup_steps):
            print_(f'WARMUP STEP {warmup_step}', 'green')
            for agent_idx, _ in enumerate(self._agents):
                final_summary = self._run(agent_idx=agent_idx, period='warmup')
                self._agents_warmup_score[agent_idx] += np.sum(final_summary.total_score) / self._lc.batch_size
        
        self._save_diversity_filter(name='results_warmup_period.csv'), self._reset_diversity_filter()
        probs = softmax(self._agents_warmup_score)

        #####################################
        ########### ROUTING PERIOD ##########
        #####################################
        for routing_step in range(self._lc.routing_steps):
            print_(f'ROUTING STEP {routing_step}', 'blue')
            
            if routing_step == 0: mask_idx = self._router.sample_mask(p=probs)
            else : mask_idx = self._router.sample_mask()
            mask_idx = mask_idx.cpu().item() 

            final_summary = self._run(agent_idx=mask_idx, period='routing')

            # Update router-learner
            p = self._update_router(final_summary, mask_idx, routing_step)
            self._router_accumulated_p += p 

            # Visualization
            if routing_step % 10 == 0 :
                self._visualize_router(p, step=routing_step)

            clean_memory(final_summary, mask_idx, p)

        clean_memory(self._router_accumulated_p, self._router_ps, self._router_argmax_idx)
        self._save_diversity_filter(name='results_routing_period.csv'), self._reset_diversity_filter()
        exit()
        # Get top agents and free memory of others
        top_agents_idx = self._get_topk_agent_idx(k=self._lc.num_mutations)
        self._free_agents(keep=top_agents_idx)

        #####################################
        ########## EVOLVING PERIOD ##########
        #####################################

        self._init_evolve()

        for evolve_step in range(self._lc.evolving_steps):
            print_(f'EVOLVE STEP {evolve_step}')
            input_sequences = [x[1] for x in self._topk_list]
            for agent_idx in top_agents_idx: 
                if agent_idx == top_agents_idx[0]:
                    self._topk_list = []
                self._evolve(input_sequences, agent_idx, top_agents_idx, period='evolve')

            self._save_diversity_filter(name=f'results_evolving_step{evolve_step}.csv')

        self._save_diversity_filter(name=f'results_evolving_step{evolve_step}.csv')
        self._logger._summary_writer.close()

    def _create_report(self, agent_idx, dtos, final_summary) -> List[OutputDTO]:
        outputs = [dto.output for dto in dtos]
        aminoacids = [self._chemistry.get_generated_amino_acids(output) for output in outputs]
        report = [OutputDTO(agent_idx=agent_idx, peptide=smi, amino_acids=aa) for smi, aa in zip(final_summary.scored_smiles, aminoacids)]
        return report

    def to_tensor(self, tensor):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        if torch.cuda.is_available():
            return torch.autograd.Variable(tensor).cuda()
        return torch.autograd.Variable(tensor)

    @torch.no_grad()
    def _get_distance_to_prior(self, prior_likelihood: Union[torch.Tensor, np.ndarray],
                              distance_threshold=-20.) -> np.ndarray:
        """prior_likelihood and distance_threshold have negative values"""
        if type(prior_likelihood) == torch.Tensor:
            ones = torch.ones_like(prior_likelihood, requires_grad=False)
            mask = torch.where(prior_likelihood > distance_threshold, ones, distance_threshold / prior_likelihood)
            mask = mask.cpu().numpy()
        else:
            ones = np.ones_like(prior_likelihood)
            mask = np.where(prior_likelihood > distance_threshold, ones, distance_threshold / prior_likelihood)
        return mask

    def _get_indices_of_unique_smiles(self, smiles: [str]) -> np.array:
        smiles_list = []
        for indx, smile in enumerate(smiles):
            if self._conversions.smile_to_mol(smile) is not None:
                canonical_smiles = self._chemistry.canonicalize_smiles([smile], isomericSmiles=True)
                smiles_list.append(canonical_smiles[0])
            else:
                smiles_list.append(f"{indx}_INVALID")
        _, idxs = np.unique(np.array(smiles_list), return_index=True)
        sorted_indices = np.sort(idxs)
        return sorted_indices

    def _keep_unique(self, scoring_input: ScoringInputDTO, unique_indices, sampled_sequences_dto: List[SampledSequencesDTO]) -> Tuple[ScoringInputDTO, List[SampledSequencesDTO]]:
        scoring_input_unique = copy.deepcopy(scoring_input)
        scoring_input_unique.peptide_outputs = [scoring_input.peptide_outputs[idx] for idx in unique_indices]
        scoring_input_unique.peptides = [scoring_input.peptides[idx] for idx in unique_indices]
        scoring_input_unique.chuckles = [scoring_input.chuckles[idx] for idx in unique_indices]
        sampled_sequences_dto_unique = [sampled_sequences_dto[idx] for idx in unique_indices]
        return scoring_input_unique, sampled_sequences_dto_unique

    def _spawn_agents(self, agent, num_agents) : 
        # Move network models to CPU before copying to save GPU memory
        if torch.cuda.is_available():
            agent.generative_model.network.cpu()
            self._prior.generative_model.network.cpu()

        agents = [copy.deepcopy(agent) for _ in range(num_agents)]

        # Move agents back to GPU one at a time
        if torch.cuda.is_available():
            for agent in agents:
                agent.generative_model.network.cuda()
                agent.generative_model.device = torch.device('cuda')
                self._prior.generative_model.network.cuda()
                self._prior.generative_model.device = torch.device('cuda')
        
        agents_optimizer = [torch.optim.Adam(agent.generative_model.get_network_parameters(), lr=self._lc.learning_rate) for agent in agents]
        agents_baseline = np.zeros(num_agents)
        agents_warmup_score = [0 for _ in range(num_agents)]   
        return agents, agents_baseline, agents_optimizer, agents_warmup_score
        
    def _spawn_router(self, num_agents) : 
        router = Router(num_agents)
        router_optimizer = torch.optim.Adam(router.parameters(), lr=self._lc.router_learning_rate)
        router_anneal_steps = int(self._lc.router_entropy_frac * self._lc.routing_steps)
        router_baseline = 0.0
        router_accumulated_p = np.zeros(num_agents)
        router_ps, router_img, router_argmax_idx = [], [], [] 
        return router, router_baseline, router_anneal_steps, router_optimizer, router_accumulated_p, router_ps, router_img, router_argmax_idx

    def _update_topk_list(self, final_summary: FinalSummary, k: int): 
        for chuckle, score in zip(final_summary.input_chuckles, final_summary.total_score):
            # Skip duplicates (based on chuckle text)
            if chuckle in (c for _, c in self._topk_list):
                continue

            # Push to heap if fewer than 16 items
            if len(self._topk_list) < k:
                heapq.heappush(self._topk_list, (score, chuckle))
            else:
                # Replace smallest if the new one is higher
                if score > self._topk_list[0][0]:
                    heapq.heapreplace(self._topk_list, (score, chuckle))

    def _get_topk_agent_idx(self, k: int = 2) -> List[int]:
        """Get indices of top-k agents based on their probabilities from router-learner"""
        if np.all(self._router_accumulated_p == 0) : 
            return self._lc.evolve_positions
        p = self._router_accumulated_p
        topk_indices = np.argsort(p)[-k:][::-1]
        topk_indices.sort()
        return topk_indices.tolist()

    def _create_sampled_sequences_dto(self, monomer_combos: List[str], mask_chuckles: str) -> List[SampledSequencesDTO] : 
        """Create DTO objects for sampled sequences with masked input and monomer outputs."""
        dtos = [] 
        for monomers in monomer_combos : 
            dtos.append(SampledSequencesDTO(input=mask_chuckles, output=monomers, nll=-1))
        return dtos

    def _visualize_router(self, p: np.ndarray, step: int):
        """Visualize router-learning by plotting argmax indices and animating mask distribution."""
        self._router_ps.append(p)   
        self._router_argmax_idx.append(int(p.argmax()))
        self._router_img.append(get_mask_smiles_visualization(self._config.input_sequence, p, step))
        self._router_img[0].save(f'{self._config.logging.logging_path}/router_smiles.gif', save_all=True, append_images=self._router_img[1:], optimize=True, duration=300, loop=0)
        plot_argmax_indices(self._router_argmax_idx, save_path=f'{self._config.logging.logging_path}/router_argmax.png')
        animate_router_distribution(self._router_ps, save_path=f'{self._config.logging.logging_path}/router_animation.gif')
        save_probability_histogram(self._router_accumulated_p, save_path=f'{self._config.logging.logging_path}/router_accumulated_probs.png')
        save_probability_histogram(softmax(self._router_accumulated_p), save_path=f'{self._config.logging.logging_path}/router_accumulated_probs_softmax.png')

    def _save_diversity_filter(self, name) : 
        self._logger.save_final_state(self._diversity_filter.get_memory_as_dataframe(), name=name)

    def _reset_diversity_filter(self) : 
        clean_memory(self._diversity_filter)
        self._diversity_filter = DiversityFilter(self._config.diversity_filter) 

    def _free_agents(self, keep: List[int]):
        for i in range(len(self._agents)):
            if i not in keep:
                if self._agents[i] is not None:
                    self._agents[i].generative_model.network.cpu()
                    self._agents[i] = None
                    self._agents_optimizer[i] = None
        clean_memory()
