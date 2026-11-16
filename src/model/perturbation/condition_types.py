"""Shared outputs for pathway condition encoders."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class ConditionEncoding:
    """Condition tokens plus optional prototype-specific Flow context."""

    condition_tokens: Tensor
    pathway_context: Tensor | None
    activities: dict[str, Tensor]
    diagnostics: dict[str, Tensor]

    def __iter__(self):
        """Keep legacy tuple consumers working during the interface migration."""
        yield self.condition_tokens
        yield self.activities["selected"]
        yield self.diagnostics["individual"]
