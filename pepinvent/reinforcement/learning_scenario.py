import copy
import time
import numpy as np
import torch
import torch.utils.data as tud
from heapq import nlargest
from typing import List, Tuple, Union
from reinvent_chemistry import Conversions
from pepinvent.reinforcement.utils import softmax, combos, print_, plot_argmax_indices, animate_router_distribution, save_probability_histogram, mask_chuckles, fill_chuckles, get_mask_smiles_visualization
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
        self._total_steps = (
              (self._lc.warmup_steps * self._num_agents) 
            + (self._lc.routing_steps)
            + (self._lc.evolving_steps * self._lc.optimizing_steps_before_evolve * self._lc.num_mutations)
        )
        
        (
            self._agents,
            self._agents_bucket,
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

    def _sample(self, agent_idx: int, input_sequence: str) -> List[SampledSequencesDTO]:
        agent = self._agents[agent_idx]
        input = self._lc.batch_size * [input_sequence]

        dataset = Dataset(input, vocabulary=agent.vocabulary, tokenizer=agent.tokenizer)
        data_loader = tud.DataLoader(
            dataset, self._lc.batch_size, shuffle=False, collate_fn=Dataset.collate_fn
        )
        
        with torch.no_grad():
            for batch in data_loader:
                src, src_mask = batch
                sequence_dtos = agent.sample(src, src_mask)
                clean_memory(src, src_mask)
                return sequence_dtos

    def _score(self, agent_idx: int, input_sequence: str, sampled_sequences_dto: List[SampledSequencesDTO], step: int, period: str) -> Tuple[FinalSummary, List[SampledSequencesDTO]]:
        outputs = [dto.output for dto in sampled_sequences_dto]
        peptides = [self._chemistry.fill_source_peptide(input_sequence, output) for output in outputs]

        scoring_input = ScoringInputDTO(peptides=peptides, peptide_input=input_sequence,
                                        peptide_outputs=outputs)
        
        # uniqueness check
        sorted_indices = self._get_indices_of_unique_smiles(scoring_input.peptides)
        scoring_input_unique, sampled_sequences_dto_unique = self._keep_unique(scoring_input, sorted_indices, sampled_sequences_dto)
        print(f" unique molecules:{len(scoring_input_unique.peptides)}\n")
        final_summary = self._scoring_function.calculate_score(scoring_input)
        final_summary.monomers = [dto.output for dto in sampled_sequences_dto_unique]
        final_summary.total_score = self._diversity_filter.update_score(final_summary, sampled_sequences_dto_unique, step, period, mask_pos=agent_idx)
        return final_summary, sampled_sequences_dto_unique

    def _update(self, agent_idx: int, final_summary: FinalSummary, sampled_sequences_dto: List[SampledSequencesDTO]) -> LikelihoodDTO:
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

        score_tensor = score_tensor - self._agents_baseline[agent_idx]
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

        dtos = self._sample(agent_idx, input_sequence) 
        final_summary, dtos = self._score(agent_idx, input_sequence, dtos, self._global_step, period)
        likelihoods = self._update(agent_idx, final_summary, dtos)

        report = self._create_report(agent_idx, dtos, final_summary)
        self._log(agent_idx, self._start_time, self._global_step, final_summary, likelihoods, report)

        self._update_agents_bucket(agent_idx, final_summary)
        
        clean_memory(dtos, likelihoods, report)

        self._global_step += 1

        return final_summary

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
        
        # Get top agents and free memory of others
        top_agents_idx = self._get_topk_agent_idx(k=self._lc.num_mutations)
        self._free_agents(keep=top_agents_idx)

        #####################################
        ########## EVOLVING PERIOD ##########
        #####################################
        self._config.input_sequence = self._evolve(top_agents_idx, step=0)
        for evolve_step in range(self._lc.evolving_steps):
            print_(f'EVOLVE STEP {evolve_step}')
            for optimize_step in range(self._lc.optimizing_steps_before_evolve):
                print_(f'OPTIMIZE STEP {optimize_step}', 'magenta')
                for agent_idx in top_agents_idx:
                    final_summary = self._run(agent_idx=agent_idx, period='evolving')

            self._config.input_sequence = self._evolve(top_agents_idx, evolve_step)

        self._save_diversity_filter(name=f'results_evolving_step{evolve_step}.csv')
        self._logger._summary_writer.close()


    def _evolve(self, top_agents_idx, step) : 
        input_sequence = mask_chuckles(self._config.input_sequence, top_agents_idx)
        top_agents_bucket = [self._agents_bucket[i] for i in top_agents_idx] 
        monomers_combo = combos(top_agents_bucket)
        dtos = self._create_sampled_sequences_dto(monomers_combo, input_sequence) 
        tmp_diversity_filter = self._diversity_filter
        self._reset_diversity_filter()
        final_summary, dtos = self._score(top_agents_idx, input_sequence, dtos, self._global_step, period='evolving') 
        self._diversity_filter = tmp_diversity_filter
        
        best_score, best_monomers = 0, None 

        with open(f'{self._config.logging.result_path}/evolving_combinations.txt', 'a') as f:
            for idx in final_summary.valid_idxs:
                score, monomers = final_summary.total_score[idx], final_summary.monomers[idx]
                print(f'Score: {score:.4f}, Monomers: {monomers}')
                f.write(f'Score: {score}, Monomers: {monomers}\n')
                if score > best_score:
                    best_score, best_monomers = score, monomers
            f.write('=' * 100 + '\n')
        
        evolved_peptide = fill_chuckles(input_sequence, best_monomers)

        with open(f'{self._config.logging.result_path}/evolving_peptides.txt', 'a') as f:
            f.write(f'{evolved_peptide}\n')

        # If evolved peptide is different from current peptide -> Reset agents' bucket
        if evolved_peptide != self._config.input_sequence:
            print_(f'PEPTIDE HAS EVOLVED TO: {evolved_peptide}', 'green')
            self._agents_bucket = {agent_idx: [(100.0, best_monomer)] for agent_idx, best_monomer in zip(top_agents_idx, best_monomers.split('|'))}
            self._save_diversity_filter(name=f'results_evolving_step{step}.csv'), self._reset_diversity_filter()
            
        return evolved_peptide

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
        scoring_input.peptide_outputs = [scoring_input.peptide_outputs[idx] for idx in unique_indices]
        scoring_input.peptides = [scoring_input.peptides[idx] for idx in unique_indices]
        sampled_sequences_dto = [sampled_sequences_dto[idx] for idx in unique_indices]
        return scoring_input, sampled_sequences_dto
    

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
        agents_bucket = {agent_idx: [(100.0, input_monomer)] for agent_idx, input_monomer in enumerate(self._input_monomers)}
        agents_baseline = np.zeros(num_agents)
        agents_warmup_score = [0 for _ in range(num_agents)]   
        return agents, agents_bucket, agents_baseline, agents_optimizer, agents_warmup_score
        
    def _spawn_router(self, num_agents) : 
        router = Router(num_agents)
        router_optimizer = torch.optim.Adam(router.parameters(), lr=self._lc.router_learning_rate)
        router_anneal_steps = int(self._lc.router_entropy_frac * self._lc.routing_steps)
        router_baseline = 0.0
        router_accumulated_p = np.zeros(num_agents)
        router_ps, router_img, router_argmax_idx = [], [], [] 
        return router, router_baseline, router_anneal_steps, router_optimizer, router_accumulated_p, router_ps, router_img, router_argmax_idx

    def _update_agents_bucket(self, agent_idx, final_summary: FinalSummary):
        """Add a tuple (float, string) to dictionary value list keeping top 10 by float."""
        def _add_tuple(d, key, new_tuple):
            if key not in d:
                d[key] = []
            d[key].append(new_tuple)
            d[key] = nlargest(self._lc.num_top_monomers + 1, d[key], key=lambda x: x[0])

        scores, monomers, valid_idxs = final_summary.total_score, final_summary.monomers, final_summary.valid_idxs
        for idx in valid_idxs:
            _add_tuple(self._agents_bucket, agent_idx, (scores[idx], monomers[idx]))

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
