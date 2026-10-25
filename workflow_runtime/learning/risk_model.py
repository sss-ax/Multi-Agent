"""Two-head risk network: Verify(s) and policy-conditional Value(s, pi)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DualRiskPrediction:
    probability_current_correct: float
    probability_final_success: float


def build_dual_head_network(
    state_dim: int,
    policy_dim: int,
    *,
    hidden_dim: int = 128,
) -> Any:
    """Build lazily so data tooling does not require importing torch."""
    import torch

    if state_dim <= 0 or policy_dim <= 0 or hidden_dim <= 0:
        raise ValueError("dual-head dimensions must be positive")

    class DualHeadRiskNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.state_encoder = torch.nn.Sequential(
                torch.nn.Linear(state_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
            )
            # Verify Head cannot access policy features by construction.
            self.verify_head = torch.nn.Linear(hidden_dim, 1)
            self.value_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim + policy_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 1),
            )

        def forward(self, state_features, policy_features):
            encoded_state = self.state_encoder(state_features)
            current_correct_logit = self.verify_head(encoded_state).squeeze(-1)
            final_success_logit = self.value_head(
                torch.cat((encoded_state, policy_features), dim=-1)
            ).squeeze(-1)
            return current_correct_logit, final_success_logit

        def predict_proba(self, state_features, policy_features):
            current_logit, success_logit = self.forward(state_features, policy_features)
            return torch.sigmoid(current_logit), torch.sigmoid(success_logit)

    return DualHeadRiskNetwork()


def masked_dual_head_bce_loss(
    current_correct_logits: Any,
    final_success_logits: Any,
    current_correct_targets: Any,
    final_success_targets: Any,
) -> Any:
    """Train either head when the other oracle label is unavailable.

    Missing labels are represented by NaN. At least one label in the batch is
    required, preventing an apparently successful zero-signal update.
    """
    import torch
    import torch.nn.functional as functional

    current_mask = torch.isfinite(current_correct_targets)
    success_mask = torch.isfinite(final_success_targets)
    losses = []
    if bool(current_mask.any()):
        losses.append(functional.binary_cross_entropy_with_logits(
            current_correct_logits[current_mask],
            current_correct_targets[current_mask].float(),
        ))
    if bool(success_mask.any()):
        losses.append(functional.binary_cross_entropy_with_logits(
            final_success_logits[success_mask],
            final_success_targets[success_mask].float(),
        ))
    if not losses:
        raise ValueError("dual-head batch contains no finite labels")
    return sum(losses) / len(losses)
