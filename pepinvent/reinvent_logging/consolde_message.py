import time
import numpy as np

from pepinvent.reinvent_logging.logging_utils import fraction_valid_smiles
from pepinvent.scoring_function.score_summary import FinalSummary

class ConsoleMessage:

    def create(self, agent_idx, start_time, n_steps, step, smiles,
               mean_score, score_summary: FinalSummary, score,
               agent_likelihood, prior_likelihood, augmented_likelihood):
        time_message = self._time_progress(start_time, n_steps, step, smiles, mean_score)
        score_message = self._score_profile(agent_idx, score_summary.scored_smiles, score_summary.monomers, agent_likelihood, prior_likelihood,
                                            augmented_likelihood, score)
        score_breakdown = self._score_summary_breakdown(score_summary)
        message = time_message + score_message + score_breakdown
        return message

    def _time_progress(self, start_time, n_steps, step, smiles, mean_score):
        time_elapsed = int(time.time() - start_time)
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        valid_fraction = fraction_valid_smiles(smiles)
        message = (f"\n Step {step}   Fraction valid SMILES: {valid_fraction:4.1f}   Score: {mean_score:.4f}   "
                   f"Time elapsed: {time_elapsed}   "
                   f"Time left: {time_left:.1f}\n")
        return message


    def _score_profile(self, agent_idx, smiles, monomers, agent_likelihood, prior_likelihood, augmented_likelihood, score):
        # Convert inputs to numpy arrays if needed
        def to_numpy(x):
            if hasattr(x, 'data') and hasattr(x.data, 'cpu'):
                # torch tensor
                return x.data.cpu().numpy()
            else:
                # likely already numpy or memoryview
                return np.asarray(x)
        augmented_likelihood = to_numpy(augmented_likelihood)
        agent_likelihood = to_numpy(agent_likelihood)
        prior_likelihood = to_numpy(prior_likelihood)
        score = to_numpy(score)

        combined = list(zip(score, smiles, monomers, agent_likelihood, prior_likelihood, augmented_likelihood))
        combined.sort(key=lambda x: x[0], reverse=True)
        message = "     ".join(["  Agent", "Prior", "Target", "Score"] + ["SMILES\n"])
        for i in range(min(10, len(combined))):
            s, smi, monomer, a_lik, p_lik, aug_lik = combined[i]
            message += f'{a_lik:6.2f}    {p_lik:6.2f}    {aug_lik:6.2f}    {s:6.2f} '
            message += f"     {smi}\n"
        return message



    def _score_summary_breakdown(self, score_summary: FinalSummary):
        # Extract the number of scored SMILES and score components
        num_scores = len(score_summary.scored_smiles)
        num_components = len(score_summary.profile)

        # Assume the first component contains the total score used for sorting
        total_scores = score_summary.profile[0].score

        # Create sorted indices based on descending total score
        sorted_indices = sorted(range(num_scores), key=lambda i: total_scores[i], reverse=True)

        # Header row
        message = "   ".join([c.name for c in score_summary.profile])
        message += "\n"

        # Print scores in sorted order
        for idx in sorted_indices[:10]:
            for summary in score_summary.profile:
                message += f"{summary.score[idx]:.4f}   "
            message += "\n"

        return message
