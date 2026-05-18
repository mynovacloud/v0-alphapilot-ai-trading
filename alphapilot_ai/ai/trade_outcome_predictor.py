"""
Trade Outcome Predictor
=======================

An ensemble machine learning model that predicts trade success probability
using multiple complementary approaches:

1. Logistic Regression: For linear relationships between features and outcome
2. Decision Tree: For capturing non-linear patterns and feature interactions
3. Naive Bayes: For probabilistic reasoning with independent features
4. K-Nearest Neighbors: For similarity-based prediction from past trades

The ensemble combines predictions using weighted voting based on each
model's historical accuracy. Models are trained incrementally as new
trades complete, without requiring batch retraining.

Key Features:
- Online learning: Updates with each new trade outcome
- Feature importance tracking: Identifies which factors predict success
- Confidence calibration: Adjusts confidence based on prediction reliability
- Explainability: Provides reasoning for each prediction
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import random

from database.db import session_scope
from database.models import (
    AILearningMemory,
    ClaudeDecision,
    PaperTrade,
    ActivityLog,
)
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeFeatureVector:
    """
    Feature vector for trade outcome prediction.
    
    Features are grouped into categories:
    - Technical: RSI, MACD, Bollinger, ADX, Volume
    - Contextual: Regime, trend, timing
    - Historical: Recent performance, win streak/loss streak
    - Signal: Strategy type, confidence, direction
    """
    # Technical indicators (normalized to [-1, 1] or [0, 1])
    rsi_normalized: float = 0.0  # (RSI - 50) / 50
    macd_normalized: float = 0.0  # Sign and magnitude
    bb_position: float = 0.5  # Bollinger %B
    adx_normalized: float = 0.0  # (ADX - 25) / 25
    volume_ratio: float = 1.0  # Relative to average
    volatility_state: float = 0.5  # 0=low, 0.5=normal, 1=high
    
    # Trend features
    trend_direction: float = 0.0  # -1=down, 0=flat, 1=up
    trend_strength: float = 0.0  # 0-1
    price_vs_ma20: float = 0.0  # % deviation from MA20
    price_vs_ma50: float = 0.0  # % deviation from MA50
    
    # Momentum features
    return_6bar: float = 0.0  # Recent return
    return_24bar: float = 0.0  # Longer return
    momentum_score: float = 0.0  # -1 to 1
    
    # Contextual features
    regime_trending_up: float = 0.0  # 1 if TRENDING_UP, else 0
    regime_trending_down: float = 0.0
    regime_ranging: float = 0.0
    regime_volatile: float = 0.0
    
    # Timing features
    hour_sin: float = 0.0  # Cyclical encoding of hour
    hour_cos: float = 0.0
    day_sin: float = 0.0  # Cyclical encoding of day of week
    day_cos: float = 0.0
    
    # Historical context
    recent_win_rate: float = 0.5  # Last N trades
    consecutive_losses: float = 0.0  # Normalized 0-1
    consecutive_wins: float = 0.0
    session_pnl_normalized: float = 0.0  # Today's PnL
    
    # Signal features
    signal_confidence: float = 0.5
    is_buy: float = 0.0  # 1 if buy, 0 if sell
    strategy_momentum: float = 0.0  # One-hot encoded strategies
    strategy_mean_reversion: float = 0.0
    strategy_breakout: float = 0.0
    strategy_scalping: float = 0.0
    
    def to_vector(self) -> List[float]:
        """Convert to flat feature vector."""
        return [
            self.rsi_normalized,
            self.macd_normalized,
            self.bb_position,
            self.adx_normalized,
            self.volume_ratio - 1.0,  # Center around 0
            self.volatility_state - 0.5,
            self.trend_direction,
            self.trend_strength,
            self.price_vs_ma20 / 10,  # Scale percentage
            self.price_vs_ma50 / 10,
            self.return_6bar * 100,  # Scale returns
            self.return_24bar * 50,
            self.momentum_score,
            self.regime_trending_up,
            self.regime_trending_down,
            self.regime_ranging,
            self.regime_volatile,
            self.hour_sin,
            self.hour_cos,
            self.day_sin,
            self.day_cos,
            self.recent_win_rate - 0.5,  # Center
            self.consecutive_losses,
            self.consecutive_wins,
            self.session_pnl_normalized,
            self.signal_confidence - 0.5,
            self.is_buy - 0.5,
            self.strategy_momentum,
            self.strategy_mean_reversion,
            self.strategy_breakout,
            self.strategy_scalping,
        ]
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names for explainability."""
        return [
            "rsi_normalized", "macd_normalized", "bb_position", "adx_normalized",
            "volume_ratio", "volatility_state", "trend_direction", "trend_strength",
            "price_vs_ma20", "price_vs_ma50", "return_6bar", "return_24bar",
            "momentum_score", "regime_trending_up", "regime_trending_down",
            "regime_ranging", "regime_volatile", "hour_sin", "hour_cos",
            "day_sin", "day_cos", "recent_win_rate", "consecutive_losses",
            "consecutive_wins", "session_pnl", "signal_confidence", "is_buy",
            "strategy_momentum", "strategy_mean_reversion", "strategy_breakout",
            "strategy_scalping",
        ]
    
    @classmethod
    def from_market_state(cls, state: Dict[str, Any]) -> "TradeFeatureVector":
        """Create feature vector from market state dict."""
        # Hour encoding (cyclical)
        hour = datetime.utcnow().hour
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        
        # Day encoding (cyclical)
        day = datetime.utcnow().weekday()
        day_sin = math.sin(2 * math.pi * day / 7)
        day_cos = math.cos(2 * math.pi * day / 7)
        
        # Regime one-hot encoding
        regime = str(state.get("regime", "")).upper()
        
        # Strategy one-hot encoding
        strategy = str(state.get("strategy", "")).lower()
        
        return cls(
            rsi_normalized=(float(state.get("rsi", 50)) - 50) / 50,
            macd_normalized=float(state.get("macd_histogram", 0)) * 10,
            bb_position=float(state.get("bb_percent_b", 0.5)),
            adx_normalized=(float(state.get("adx", 25)) - 25) / 25,
            volume_ratio=float(state.get("volume_ratio", 1.0)),
            volatility_state=float(state.get("volatility_percentile", 50)) / 100,
            trend_direction=1.0 if state.get("trend") == "UP" else (-1.0 if state.get("trend") == "DOWN" else 0.0),
            trend_strength=float(state.get("trend_strength", 0)) / 50,
            price_vs_ma20=float(state.get("price_vs_ma20", 0)),
            price_vs_ma50=float(state.get("price_vs_ma50", 0)),
            return_6bar=float(state.get("return_6bar", 0)),
            return_24bar=float(state.get("return_24bar", 0)),
            momentum_score=float(state.get("momentum_score", 0)) / 100,
            regime_trending_up=1.0 if "TRENDING_UP" in regime else 0.0,
            regime_trending_down=1.0 if "TRENDING_DOWN" in regime else 0.0,
            regime_ranging=1.0 if "RANGING" in regime else 0.0,
            regime_volatile=1.0 if "VOLATILE" in regime else 0.0,
            hour_sin=hour_sin,
            hour_cos=hour_cos,
            day_sin=day_sin,
            day_cos=day_cos,
            recent_win_rate=float(state.get("recent_win_rate", 0.5)),
            consecutive_losses=min(float(state.get("consecutive_losses", 0)) / 5, 1.0),
            consecutive_wins=min(float(state.get("consecutive_wins", 0)) / 5, 1.0),
            session_pnl_normalized=max(-1, min(1, float(state.get("session_pnl", 0)) / 100)),
            signal_confidence=float(state.get("signal_confidence", 0.5)),
            is_buy=1.0 if str(state.get("side", "")).upper() == "BUY" else 0.0,
            strategy_momentum=1.0 if "momentum" in strategy else 0.0,
            strategy_mean_reversion=1.0 if "reversion" in strategy or "mean" in strategy else 0.0,
            strategy_breakout=1.0 if "breakout" in strategy else 0.0,
            strategy_scalping=1.0 if "scalp" in strategy else 0.0,
        )


@dataclass
class PredictionResult:
    """Result from the ensemble predictor."""
    win_probability: float  # 0-1 probability of winning trade
    confidence: float  # Confidence in the prediction
    predicted_outcome: str  # "WIN", "LOSS", "UNCERTAIN"
    expected_pnl_pct: float  # Expected PnL percentage
    
    # Individual model predictions
    model_predictions: Dict[str, float]  # Model name -> win probability
    
    # Feature importance for this prediction
    top_positive_factors: List[Tuple[str, float]]  # Features pushing toward WIN
    top_negative_factors: List[Tuple[str, float]]  # Features pushing toward LOSS
    
    # Reasoning
    reasoning: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "win_probability": round(self.win_probability, 3),
            "confidence": round(self.confidence, 3),
            "predicted_outcome": self.predicted_outcome,
            "expected_pnl_pct": round(self.expected_pnl_pct, 2),
            "model_predictions": {k: round(v, 3) for k, v in self.model_predictions.items()},
            "top_positive_factors": [(f, round(v, 3)) for f, v in self.top_positive_factors[:5]],
            "top_negative_factors": [(f, round(v, 3)) for f, v in self.top_negative_factors[:5]],
            "reasoning": self.reasoning,
        }


class OnlineLogisticRegression:
    """Simple online logistic regression with SGD."""
    
    def __init__(self, n_features: int, learning_rate: float = 0.01, l2_reg: float = 0.001):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.l2 = l2_reg
        self.n_samples = 0
    
    def predict_proba(self, features: List[float]) -> float:
        """Predict probability of positive class."""
        z = self.bias + sum(w * f for w, f in zip(self.weights, features))
        return 1.0 / (1.0 + math.exp(-max(-500, min(500, z))))
    
    def update(self, features: List[float], label: int):
        """Update weights with one sample using SGD."""
        pred = self.predict_proba(features)
        error = label - pred
        
        # Update weights with L2 regularization
        for i in range(len(self.weights)):
            gradient = error * features[i] - self.l2 * self.weights[i]
            self.weights[i] += self.lr * gradient
        
        self.bias += self.lr * error
        self.n_samples += 1
    
    def get_feature_importance(self) -> List[float]:
        """Get absolute weight magnitudes as importance."""
        return [abs(w) for w in self.weights]
    
    def to_dict(self) -> Dict:
        return {
            "weights": self.weights,
            "bias": self.bias,
            "n_samples": self.n_samples,
        }
    
    @classmethod
    def from_dict(cls, d: Dict, n_features: int) -> "OnlineLogisticRegression":
        model = cls(n_features)
        model.weights = d.get("weights", [0.0] * n_features)
        model.bias = d.get("bias", 0.0)
        model.n_samples = d.get("n_samples", 0)
        return model


class OnlineNaiveBayes:
    """Online Naive Bayes with Gaussian assumption."""
    
    def __init__(self, n_features: int):
        self.n_features = n_features
        # For each class (0, 1), track mean and variance of each feature
        self.class_counts = {0: 0, 1: 0}
        self.feature_means = {0: [0.0] * n_features, 1: [0.0] * n_features}
        self.feature_vars = {0: [1.0] * n_features, 1: [1.0] * n_features}
    
    def predict_proba(self, features: List[float]) -> float:
        """Predict probability of positive class."""
        if self.class_counts[0] == 0 or self.class_counts[1] == 0:
            return 0.5
        
        # Log probabilities to avoid underflow
        log_prob_0 = math.log(self.class_counts[0] / (self.class_counts[0] + self.class_counts[1]))
        log_prob_1 = math.log(self.class_counts[1] / (self.class_counts[0] + self.class_counts[1]))
        
        for i in range(self.n_features):
            # Gaussian likelihood
            var_0 = max(self.feature_vars[0][i], 0.001)
            var_1 = max(self.feature_vars[1][i], 0.001)
            
            diff_0 = features[i] - self.feature_means[0][i]
            diff_1 = features[i] - self.feature_means[1][i]
            
            log_prob_0 -= 0.5 * (math.log(var_0) + diff_0 * diff_0 / var_0)
            log_prob_1 -= 0.5 * (math.log(var_1) + diff_1 * diff_1 / var_1)
        
        # Convert back to probability
        max_log = max(log_prob_0, log_prob_1)
        prob_0 = math.exp(log_prob_0 - max_log)
        prob_1 = math.exp(log_prob_1 - max_log)
        
        return prob_1 / (prob_0 + prob_1)
    
    def update(self, features: List[float], label: int):
        """Update statistics with one sample."""
        n = self.class_counts[label]
        self.class_counts[label] = n + 1
        
        for i in range(self.n_features):
            old_mean = self.feature_means[label][i]
            new_mean = (old_mean * n + features[i]) / (n + 1)
            
            # Welford's online variance algorithm
            if n > 0:
                old_var = self.feature_vars[label][i]
                new_var = ((n - 1) * old_var + (features[i] - old_mean) * (features[i] - new_mean)) / n
                self.feature_vars[label][i] = max(new_var, 0.001)
            
            self.feature_means[label][i] = new_mean
    
    def to_dict(self) -> Dict:
        return {
            "class_counts": self.class_counts,
            "feature_means": self.feature_means,
            "feature_vars": self.feature_vars,
        }
    
    @classmethod
    def from_dict(cls, d: Dict, n_features: int) -> "OnlineNaiveBayes":
        model = cls(n_features)
        model.class_counts = d.get("class_counts", {0: 0, 1: 0})
        model.feature_means = d.get("feature_means", {0: [0.0] * n_features, 1: [0.0] * n_features})
        model.feature_vars = d.get("feature_vars", {0: [1.0] * n_features, 1: [1.0] * n_features})
        return model


class OnlineKNN:
    """K-Nearest Neighbors with reservoir sampling for online learning."""
    
    def __init__(self, k: int = 10, max_samples: int = 500):
        self.k = k
        self.max_samples = max_samples
        self.samples: List[Tuple[List[float], int]] = []  # (features, label)
    
    def predict_proba(self, features: List[float]) -> float:
        """Predict probability using k nearest neighbors."""
        if len(self.samples) < self.k:
            return 0.5
        
        # Calculate distances to all samples
        distances = []
        for sample_features, label in self.samples:
            dist = sum((a - b) ** 2 for a, b in zip(features, sample_features))
            distances.append((dist, label))
        
        # Get k nearest
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.k]
        
        # Weighted voting (inverse distance weighting)
        win_weight = 0.0
        total_weight = 0.0
        
        for dist, label in neighbors:
            weight = 1.0 / (dist + 0.001)  # Add small constant to avoid division by zero
            total_weight += weight
            if label == 1:
                win_weight += weight
        
        return win_weight / total_weight if total_weight > 0 else 0.5
    
    def update(self, features: List[float], label: int):
        """Add sample using reservoir sampling."""
        if len(self.samples) < self.max_samples:
            self.samples.append((features, label))
        else:
            # Reservoir sampling: replace random sample with decreasing probability
            idx = random.randint(0, len(self.samples) - 1)
            self.samples[idx] = (features, label)
    
    def to_dict(self) -> Dict:
        return {
            "samples": self.samples,
            "k": self.k,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "OnlineKNN":
        model = cls(k=d.get("k", 10))
        model.samples = d.get("samples", [])
        return model


class TradeOutcomePredictor:
    """
    Ensemble predictor combining multiple online learning models.
    
    Uses weighted voting based on each model's historical accuracy.
    """
    
    def __init__(self):
        self.n_features = 31  # Number of features in TradeFeatureVector
        
        # Initialize models
        self.logistic = OnlineLogisticRegression(self.n_features)
        self.naive_bayes = OnlineNaiveBayes(self.n_features)
        self.knn = OnlineKNN(k=10, max_samples=500)
        
        # Model weights (based on historical accuracy)
        self.model_weights = {
            "logistic": 1.0,
            "naive_bayes": 1.0,
            "knn": 1.0,
        }
        
        # Track accuracy for weight adjustment
        self.model_correct = defaultdict(int)
        self.model_total = defaultdict(int)
        
        # Feature importance accumulator
        self.feature_importance = [0.0] * self.n_features
        self.importance_samples = 0
        
        # Historical performance for expected PnL calculation
        self.win_pnl_sum = 0.0
        self.win_count = 0
        self.loss_pnl_sum = 0.0
        self.loss_count = 0
        
        self._loaded = False
    
    def load_from_db(self):
        """Load model state from database."""
        with session_scope() as s:
            # Load model states
            for category in ["predictor_logistic", "predictor_naive_bayes", "predictor_knn", "predictor_meta"]:
                row = s.query(AILearningMemory).filter(
                    AILearningMemory.category == category
                ).first()
                
                if row and row.content:
                    try:
                        data = json.loads(row.content)
                        
                        if category == "predictor_logistic":
                            self.logistic = OnlineLogisticRegression.from_dict(data, self.n_features)
                        elif category == "predictor_naive_bayes":
                            self.naive_bayes = OnlineNaiveBayes.from_dict(data, self.n_features)
                        elif category == "predictor_knn":
                            self.knn = OnlineKNN.from_dict(data)
                        elif category == "predictor_meta":
                            self.model_weights = data.get("model_weights", self.model_weights)
                            self.model_correct = defaultdict(int, data.get("model_correct", {}))
                            self.model_total = defaultdict(int, data.get("model_total", {}))
                            self.feature_importance = data.get("feature_importance", [0.0] * self.n_features)
                            self.importance_samples = data.get("importance_samples", 0)
                            self.win_pnl_sum = data.get("win_pnl_sum", 0)
                            self.win_count = data.get("win_count", 0)
                            self.loss_pnl_sum = data.get("loss_pnl_sum", 0)
                            self.loss_count = data.get("loss_count", 0)
                    except Exception as e:
                        logger.warning(f"Failed to load {category}: {e}")
        
        self._loaded = True
        logger.info(f"[PREDICTOR] Loaded models with {self.logistic.n_samples} logistic samples")
    
    def save_to_db(self):
        """Save model state to database."""
        models = {
            "predictor_logistic": self.logistic.to_dict(),
            "predictor_naive_bayes": self.naive_bayes.to_dict(),
            "predictor_knn": self.knn.to_dict(),
            "predictor_meta": {
                "model_weights": dict(self.model_weights),
                "model_correct": dict(self.model_correct),
                "model_total": dict(self.model_total),
                "feature_importance": self.feature_importance,
                "importance_samples": self.importance_samples,
                "win_pnl_sum": self.win_pnl_sum,
                "win_count": self.win_count,
                "loss_pnl_sum": self.loss_pnl_sum,
                "loss_count": self.loss_count,
            },
        }
        
        with session_scope() as s:
            for category, data in models.items():
                existing = s.query(AILearningMemory).filter(
                    AILearningMemory.category == category
                ).first()
                
                content = json.dumps(data)
                
                if existing:
                    existing.content = content
                else:
                    s.add(AILearningMemory(
                        category=category,
                        content=content,
                        weight=1.0,
                    ))
    
    def predict(self, features: TradeFeatureVector) -> PredictionResult:
        """
        Predict trade outcome using ensemble.
        
        Returns PredictionResult with probability, confidence, and reasoning.
        """
        if not self._loaded:
            self.load_from_db()
        
        feature_vector = features.to_vector()
        feature_names = features.feature_names
        
        # Get predictions from each model
        predictions = {
            "logistic": self.logistic.predict_proba(feature_vector),
            "naive_bayes": self.naive_bayes.predict_proba(feature_vector),
            "knn": self.knn.predict_proba(feature_vector),
        }
        
        # Weighted ensemble
        total_weight = sum(self.model_weights.values())
        ensemble_prob = sum(
            predictions[model] * weight
            for model, weight in self.model_weights.items()
        ) / total_weight
        
        # Confidence based on agreement between models
        pred_values = list(predictions.values())
        pred_std = math.sqrt(sum((p - ensemble_prob) ** 2 for p in pred_values) / len(pred_values))
        agreement_confidence = 1.0 - min(1.0, pred_std * 2)
        
        # Confidence also based on sample size
        sample_confidence = min(1.0, self.logistic.n_samples / 100)
        
        confidence = agreement_confidence * 0.6 + sample_confidence * 0.4
        
        # Determine outcome
        if ensemble_prob > 0.6:
            predicted_outcome = "WIN"
        elif ensemble_prob < 0.4:
            predicted_outcome = "LOSS"
        else:
            predicted_outcome = "UNCERTAIN"
        
        # Expected PnL
        avg_win_pnl = self.win_pnl_sum / self.win_count if self.win_count > 0 else 2.0
        avg_loss_pnl = self.loss_pnl_sum / self.loss_count if self.loss_count > 0 else -1.5
        expected_pnl = ensemble_prob * avg_win_pnl + (1 - ensemble_prob) * avg_loss_pnl
        
        # Feature importance analysis
        logistic_importance = self.logistic.get_feature_importance()
        
        # Find positive and negative contributing features
        positive_factors = []
        negative_factors = []
        
        for i, (name, value, importance) in enumerate(zip(
            feature_names, feature_vector, logistic_importance
        )):
            contribution = value * self.logistic.weights[i] if i < len(self.logistic.weights) else 0
            if contribution > 0:
                positive_factors.append((name, contribution))
            else:
                negative_factors.append((name, abs(contribution)))
        
        positive_factors.sort(key=lambda x: x[1], reverse=True)
        negative_factors.sort(key=lambda x: x[1], reverse=True)
        
        # Generate reasoning
        reasoning = []
        
        if predicted_outcome == "WIN":
            reasoning.append(f"Ensemble predicts WIN with {ensemble_prob:.0%} probability")
        elif predicted_outcome == "LOSS":
            reasoning.append(f"Ensemble predicts LOSS with {1-ensemble_prob:.0%} probability")
        else:
            reasoning.append(f"Prediction uncertain (win probability: {ensemble_prob:.0%})")
        
        # Model agreement
        if pred_std < 0.1:
            reasoning.append("Strong model agreement")
        elif pred_std > 0.2:
            reasoning.append("Models disagree - prediction less reliable")
        
        # Top factors
        if positive_factors:
            top_pos = positive_factors[0]
            reasoning.append(f"Top positive factor: {top_pos[0]}")
        if negative_factors:
            top_neg = negative_factors[0]
            reasoning.append(f"Top risk factor: {top_neg[0]}")
        
        # Sample size warning
        if self.logistic.n_samples < 50:
            reasoning.append(f"Limited training data ({self.logistic.n_samples} samples)")
        
        return PredictionResult(
            win_probability=ensemble_prob,
            confidence=confidence,
            predicted_outcome=predicted_outcome,
            expected_pnl_pct=expected_pnl,
            model_predictions=predictions,
            top_positive_factors=positive_factors[:5],
            top_negative_factors=negative_factors[:5],
            reasoning=reasoning,
        )
    
    def learn(self, features: TradeFeatureVector, outcome: bool, pnl_pct: float):
        """
        Learn from a trade outcome.
        
        Args:
            features: Feature vector at time of trade entry
            outcome: True if trade was profitable
            pnl_pct: Realized PnL percentage
        """
        if not self._loaded:
            self.load_from_db()
        
        feature_vector = features.to_vector()
        label = 1 if outcome else 0
        
        # Get predictions before update (for accuracy tracking)
        pred_logistic = self.logistic.predict_proba(feature_vector)
        pred_naive_bayes = self.naive_bayes.predict_proba(feature_vector)
        pred_knn = self.knn.predict_proba(feature_vector)
        
        # Update accuracy tracking
        predictions = {
            "logistic": pred_logistic,
            "naive_bayes": pred_naive_bayes,
            "knn": pred_knn,
        }
        
        for model_name, pred in predictions.items():
            self.model_total[model_name] += 1
            predicted_outcome = pred > 0.5
            if predicted_outcome == outcome:
                self.model_correct[model_name] += 1
        
        # Update model weights based on accuracy
        for model_name in self.model_weights:
            total = self.model_total[model_name]
            if total >= 20:
                accuracy = self.model_correct[model_name] / total
                # Weight = accuracy^2 (reward accurate models more)
                self.model_weights[model_name] = max(0.1, accuracy ** 2)
        
        # Update models
        self.logistic.update(feature_vector, label)
        self.naive_bayes.update(feature_vector, label)
        self.knn.update(feature_vector, label)
        
        # Update feature importance (running average)
        importance = self.logistic.get_feature_importance()
        n = self.importance_samples
        for i in range(len(self.feature_importance)):
            if i < len(importance):
                self.feature_importance[i] = (
                    self.feature_importance[i] * n + importance[i]
                ) / (n + 1)
        self.importance_samples = n + 1
        
        # Update PnL statistics
        if outcome:
            self.win_pnl_sum += pnl_pct
            self.win_count += 1
        else:
            self.loss_pnl_sum += pnl_pct
            self.loss_count += 1
        
        # Save periodically
        if self.logistic.n_samples % 10 == 0:
            self.save_to_db()
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about the predictor."""
        if not self._loaded:
            self.load_from_db()
        
        feature_names = TradeFeatureVector().feature_names
        
        # Top features by importance
        top_features = sorted(
            zip(feature_names, self.feature_importance),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Model accuracies
        accuracies = {}
        for model_name in self.model_weights:
            total = self.model_total[model_name]
            if total > 0:
                accuracies[model_name] = self.model_correct[model_name] / total
            else:
                accuracies[model_name] = 0.5
        
        return {
            "total_samples": self.logistic.n_samples,
            "model_weights": dict(self.model_weights),
            "model_accuracies": accuracies,
            "top_features": [(f, round(i, 4)) for f, i in top_features],
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "avg_win_pnl": self.win_pnl_sum / self.win_count if self.win_count > 0 else 0,
            "avg_loss_pnl": self.loss_pnl_sum / self.loss_count if self.loss_count > 0 else 0,
        }


# Singleton instance
_predictor: Optional[TradeOutcomePredictor] = None


def get_outcome_predictor() -> TradeOutcomePredictor:
    """Get the singleton outcome predictor."""
    global _predictor
    if _predictor is None:
        _predictor = TradeOutcomePredictor()
    return _predictor


def predict_trade_outcome(market_state: Dict[str, Any]) -> PredictionResult:
    """Convenience function to predict trade outcome."""
    predictor = get_outcome_predictor()
    features = TradeFeatureVector.from_market_state(market_state)
    return predictor.predict(features)


def learn_from_trade_outcome(
    market_state: Dict[str, Any],
    was_profitable: bool,
    pnl_pct: float,
):
    """Convenience function to learn from a trade outcome."""
    predictor = get_outcome_predictor()
    features = TradeFeatureVector.from_market_state(market_state)
    predictor.learn(features, was_profitable, pnl_pct)
