"""Compare expert-training actions with actions executed by CEM policies."""

from __future__ import annotations

import numpy as np


def _finite_episodes(episodes):
    cleaned = []
    for actions in episodes:
        array = np.asarray(actions, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError("Each episode must contain actions with shape (T, 2)")
        finite = array[np.isfinite(array).all(axis=1)]
        if len(finite):
            cleaned.append(finite)
    if not cleaned:
        raise ValueError("No finite actions were provided")
    return cleaned


def _summary(episodes, low, high, central_low, central_high, edges):
    actions = np.concatenate(episodes)
    counts, _, _ = np.histogram2d(
        actions[:, 0], actions[:, 1], bins=edges
    )
    changes = [np.linalg.norm(np.diff(item, axis=0), axis=1) for item in episodes]
    changes = [item for item in changes if len(item)]
    near_bound = (actions - low <= 0.01 * (high - low)) | (
        high - actions <= 0.01 * (high - low)
    )
    outside_central = (actions < central_low) | (actions > central_high)
    outside_bounds = (actions < low) | (actions > high)
    return {
        "count": int(len(actions)),
        "episodes": len(episodes),
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "quantiles_05_50_95": np.quantile(actions, [0.05, 0.5, 0.95], axis=0).tolist(),
        "mean_magnitude": float(np.linalg.norm(actions, axis=1).mean()),
        "mean_step_change": float(
            np.concatenate(changes).mean() if changes else 0.0
        ),
        "fraction_near_expert_bounds": float(near_bound.any(axis=1).mean()),
        "fraction_outside_expert_bounds": float(outside_bounds.any(axis=1).mean()),
        "fraction_outside_expert_01_99": float(outside_central.any(axis=1).mean()),
        "histogram": counts.astype(np.int64).tolist(),
    }


def _jensen_shannon_bits(left, right):
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    left /= left.sum()
    right /= right.sum()
    midpoint = 0.5 * (left + right)
    left_mask = left > 0
    right_mask = right > 0
    divergence = 0.5 * (
        np.sum(left[left_mask] * np.log2(left[left_mask] / midpoint[left_mask]))
        + np.sum(right[right_mask] * np.log2(right[right_mask] / midpoint[right_mask]))
    )
    return float(divergence)


def compare_action_distributions(expert_episodes, baseline_episodes, candidate_episodes,
                                 *, bins=32):
    """Return JSON-safe summaries and shared 2-D histograms in raw action units.

    Inputs contain executed, not merely proposed, actions. Expert episodes
    should be the same episode split used for IDM gradient training.
    """
    if bins < 2:
        raise ValueError("bins must be at least 2")
    groups = {
        "expert_train": _finite_episodes(expert_episodes),
        "expert_cem": _finite_episodes(baseline_episodes),
        "fine_tuned_cem": _finite_episodes(candidate_episodes),
    }
    expert = np.concatenate(groups["expert_train"])
    low = expert.min(axis=0)
    high = expert.max(axis=0)
    if np.any(low >= high):
        raise ValueError("Expert actions must span both action dimensions")
    central_low, central_high = np.quantile(expert, [0.01, 0.99], axis=0)
    all_actions = np.concatenate([np.concatenate(episodes) for episodes in groups.values()])
    histogram_low = all_actions.min(axis=0)
    histogram_high = all_actions.max(axis=0)
    edges = [
        np.linspace(histogram_low[axis], histogram_high[axis], bins + 1)
        for axis in range(2)
    ]
    summaries = {
        name: _summary(episodes, low, high, central_low, central_high, edges)
        for name, episodes in groups.items()
    }
    expert_histogram = summaries["expert_train"]["histogram"]
    return {
        "units": "raw relative PushT action; environment multiplies by 100",
        "expert_reference": "IDM training episodes (90% split, seed 0)",
        "bins_per_axis": bins,
        "edges": [edge.tolist() for edge in edges],
        "expert_action_bounds": [low.tolist(), high.tolist()],
        "expert_central_01_99": [central_low.tolist(), central_high.tolist()],
        "groups": summaries,
        "jensen_shannon_bits_vs_expert": {
            name: _jensen_shannon_bits(expert_histogram, summaries[name]["histogram"])
            for name in ("expert_cem", "fine_tuned_cem")
        },
    }
