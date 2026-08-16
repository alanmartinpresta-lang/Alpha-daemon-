"""Self-directed improvement controller for Alpha.

The controller reads externally recorded performance diagnostics, identifies
generic symptoms, generates bounded code-change candidates, and ranks them.
It cannot edit the evaluator or the environment. Candidate code must be
tested in held-out environments before adoption.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Diagnostic:
    mean_reward: float
    min_reward: float
    reward_variance: float
    survival_rate: float

def propose(diag: Diagnostic):
    proposals = []
    # Alpha's observed learning was stable but had costly negative excursions.
    # Generate candidate changes that could reduce overreaction / update noise.
    if diag.min_reward < -0.05:
        proposals += [
            ("lower_learning_rate", "learning_rate", 0.12),
            ("lower_uncertainty_bonus", "uncertainty_bonus", 0.02),
            ("higher_discount", "discount", 0.96),
        ]
    if diag.reward_variance > 0.001:
        proposals.append(("slower_exploration_decay", "exploration_decay", 0.9995))
    return proposals

def select(train_scores):
    """Select the candidate with the best mean held-in reward."""
    if not train_scores:
        return None
    return max(train_scores, key=lambda x: x["mean_reward"])
