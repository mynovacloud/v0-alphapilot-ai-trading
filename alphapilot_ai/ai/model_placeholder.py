"""
Future ML model placeholder.

This module is intentionally empty of real ML code. The class signature is
stable so the rest of the codebase can switch to a real model without churn.

Suggested future implementation:
- scikit-learn classifier for BUY/SELL/HOLD
- XGBoost regressor for confidence score
- PyTorch model for sequence-aware decisions
"""
from __future__ import annotations

from typing import Any


class FutureMLModel:
    """Placeholder for a real ML model."""

    def __init__(self) -> None:
        self.is_trained = False

    def fit(self, X: Any, y: Any) -> None:
        # TODO: replace with real training
        self.is_trained = True

    def predict(self, X: Any) -> list[float]:
        # TODO: replace with real inference
        return [0.5 for _ in range(len(X) if hasattr(X, "__len__") else 1)]

    def save(self, path: str) -> None:
        # TODO: persist model weights
        pass

    def load(self, path: str) -> None:
        # TODO: load model weights
        self.is_trained = True
