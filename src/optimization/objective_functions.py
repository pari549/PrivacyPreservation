"""Balanced feature-based objectives for Jaya parameter search.

Recovery, utility and privacy are deterministic score proxies used solely for
optimization mechanics; they are not measurements and apply no transform.
"""
from __future__ import annotations
import numpy as np
from src.optimization.jaya_optimizer import normalize_bounded_weights

FEATURE_NAMES = ("variance", "gradient", "entropy", "roi_score")

def feature_diversity_score(weights: np.ndarray) -> float:
    normalized = normalize_bounded_weights(np.asarray(weights, dtype=np.float64))
    return float(1.0 - np.sum(normalized ** 2))

def block_priorities(features: list[dict], weights: np.ndarray) -> np.ndarray:
    normalized = normalize_bounded_weights(np.asarray(weights, dtype=np.float64))
    matrix = np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in features])
    return matrix @ normalized

def _score_components(vector: np.ndarray, block_count: int, features: list[dict]) -> tuple[float, float, float, float]:
    alpha, beta, weights = vector[:block_count], vector[block_count:2*block_count], vector[-4:]
    priority = block_priorities(features, weights)
    scaled = priority / (1.0 + float(np.max(priority)))
    recovery_proxy = float(np.mean(1.0 - alpha * scaled))
    utility_proxy = float(np.mean(1.0 - beta * scaled))
    privacy_proxy = float(np.mean(alpha * beta * scaled))
    diversity = feature_diversity_score(weights)
    return recovery_proxy, utility_proxy, privacy_proxy, diversity

def legacy_method1_objective(vector: np.ndarray, block_count: int, features: list[dict]) -> float:
    """Previous equal-average recovery, utility and privacy proxy objective."""
    recovery, utility, privacy, _ = _score_components(vector, block_count, features)
    return (recovery + utility + privacy) / 3.0

def balanced_multiobjective(vector: np.ndarray, block_count: int, features: list[dict]) -> float:
    """Equal-weight recovery, utility, privacy and feature-diversity score."""
    return float(np.mean(_score_components(vector, block_count, features)))

def method2_privacy_objective(vector: np.ndarray, block_count: int, features: list[dict]) -> float:
    """Privacy proxy only, with no recovery term."""
    alpha, beta, weights = vector[:block_count], vector[block_count:2*block_count], vector[-4:]
    priority = block_priorities(features, weights)
    return float(np.mean(alpha * beta * priority / (1.0 + float(np.max(priority)))))
