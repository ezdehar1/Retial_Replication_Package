#!/usr/bin/env python3
"""
Violation-only transition-aware DDQN for JNN tail-latency recovery.

Final experiment design:
1. Training and evaluation use disjoint Protect-SLO states only (RT95 > tau).
2. The DDQN is invoked only for active SLO violations; compliant online states
   retain the current configuration outside the learning policy.
3. The state remains unchanged at 13 values:
   [normalized RPM, normalized RT95, 11 current-configuration flags].
4. The reward remains configurable through alpha and beta.
5. Reconfiguration is evaluated using normalized Hamming distance, raw changed
   flags, minimum feasible distance, excess distance, and Functional Retention
   (FR).
6. FR uses five optional abstract feature groups: Media, Analytics,
   Recommendation, Breaking, and Advertisement. For each evaluation state, FR is
   the fraction of optional functionalities enabled in the current configuration
   that remain enabled in the selected configuration.

Example:
python DDQN_Conf_violation_only_FR.py \
  --alpha 0.90 --seed 45 --steps 120000 \
  --metrics Data/DataConf/Final_DS_JNN9_local_Desktop_UpdatedWed_all.csv \
  --configs Data/DataConf/ALL_Config_binary.csv \
  --startup Data/DataConf/JNN_startup_violation_only.csv \
  --evaluation Data/DataConf/JNN_evaluation_violation_only.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Optional training dependencies are imported defensively so --validate-only
# can still verify the datasets on machines where the RL environment is absent.
try:
    import gym
    import joblib
    import pfrl
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from pfrl import action_value, agents, explorers, replay_buffers
    TRAINING_IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    gym = None
    joblib = None
    pfrl = None
    torch = None
    nn = None
    F = None
    action_value = agents = explorers = replay_buffers = None
    TRAINING_IMPORT_ERROR = exc


FLAG_COLS: Tuple[str, ...] = (
    "adv1",
    "adv2",
    "analytics1",
    "analytics2",
    "breaking",
    "content1",
    "content2",
    "media1",
    "media2",
    "recommendation1",
    "recommendation2",
)

# Functional Retention (FR) follows the paper's five optional abstract features.
# Alternative implementations count as the same abstract functionality.
OPTIONAL_FEATURE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "media": ("media1", "media2"),
    "analytics": ("analytics1", "analytics2"),
    "recommendation": ("recommendation1", "recommendation2"),
    "breaking": ("breaking",),
    "advertisement": ("adv1", "adv2"),
}
TOTAL_OPTIONAL_FEATURE_GROUPS = int(len(OPTIONAL_FEATURE_GROUPS))

REQUIRED_STATE_COLUMNS: Tuple[str, ...] = (
    "config",
    "actual_rpm",
    "rt_95",
    "tau_ms",
    "trigger_label",
)

STATE_KEY_COLUMNS: Tuple[str, ...] = (
    "run_time_s",
    "timestamp",
    "config",
    "actual_rpm",
    "rt_95",
)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if pfrl is not None:
        pfrl.utils.set_random_seed(seed)


def min_max_scale(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def stable_rt_normalization(rt: float, mean: float, std: float, eps: float = 1e-8) -> float:
    """Map a z-score smoothly into [0, 1], matching the earlier implementation."""
    z = (float(rt) - float(mean)) / (float(std) + eps)
    return float((np.tanh(z) + 1.0) / 2.0)


def state_key(row: pd.Series) -> Tuple[str, ...]:
    return tuple(str(row[col]).strip() for col in STATE_KEY_COLUMNS)


def validate_inputs(
    metrics_df: pd.DataFrame,
    configs_df: pd.DataFrame,
    startup_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    explicit_tau: float | None,
) -> Tuple[float, Dict[str, object]]:
    missing_config_cols = [c for c in ("Config_ID", *FLAG_COLS) if c not in configs_df.columns]
    if missing_config_cols:
        raise ValueError(f"Configuration file is missing columns: {missing_config_cols}")

    missing_metric_cols = [c for c in ("actual_rpm", "rt_95") if c not in metrics_df.columns]
    if missing_metric_cols:
        raise ValueError(f"Metrics file is missing columns: {missing_metric_cols}")

    for name, frame in (("startup", startup_df), ("evaluation", evaluation_df)):
        missing = [c for c in REQUIRED_STATE_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} dataset is missing columns: {missing}")
        if frame.empty:
            raise ValueError(f"{name} dataset is empty.")

    action_ids = sorted(configs_df["Config_ID"].astype(int).unique().tolist())
    if len(action_ids) != len(configs_df["Config_ID"].unique()):
        raise ValueError("Configuration file contains duplicated Config_ID records.")
    if len(action_ids) != 204:
        logging.warning("Expected 204 JNN configurations, found %d.", len(action_ids))

    for col in FLAG_COLS:
        values = set(configs_df[col].dropna().astype(int).unique().tolist())
        if not values.issubset({0, 1}):
            raise ValueError(f"Configuration flag {col!r} is not binary: {sorted(values)}")

    startup_tau_values = sorted(startup_df["tau_ms"].astype(float).unique().tolist())
    evaluation_tau_values = sorted(evaluation_df["tau_ms"].astype(float).unique().tolist())
    if len(startup_tau_values) != 1:
        raise ValueError(f"Startup dataset has multiple tau values: {startup_tau_values}")
    if len(evaluation_tau_values) != 1:
        raise ValueError(f"Evaluation dataset has multiple tau values: {evaluation_tau_values}")
    if not np.isclose(startup_tau_values[0], evaluation_tau_values[0]):
        raise ValueError(
            "Startup and evaluation datasets use different tau values: "
            f"{startup_tau_values[0]} vs {evaluation_tau_values[0]}"
        )

    dataset_tau = float(startup_tau_values[0])
    tau_ms = float(explicit_tau) if explicit_tau is not None else dataset_tau

    if explicit_tau is not None and not np.isclose(tau_ms, dataset_tau):
        logging.warning(
            "Explicit tau %.4f differs from the derived-dataset tau %.4f. "
            "Labels will be interpreted using the explicit tau.",
            tau_ms,
            dataset_tau,
        )

    # This focused recovery experiment must contain active Protect-SLO states only.
    startup_nonviolations = int((startup_df["rt_95"].astype(float) <= tau_ms).sum())
    evaluation_nonviolations = int((evaluation_df["rt_95"].astype(float) <= tau_ms).sum())
    if startup_nonviolations or evaluation_nonviolations:
        raise ValueError(
            "Violation-only experiment received compliant rows: "
            f"startup={startup_nonviolations}, evaluation={evaluation_nonviolations}. "
            "Use the generated violation-only datasets."
        )

    startup_keys = {state_key(row) for _, row in startup_df.iterrows()}
    evaluation_keys = {state_key(row) for _, row in evaluation_df.iterrows()}
    overlap_count = len(startup_keys.intersection(evaluation_keys))
    if overlap_count:
        raise ValueError(
            f"Startup and evaluation datasets overlap in {overlap_count} exact states. "
            "Use the generated disjoint files."
        )

    report: Dict[str, object] = {
        "tau_ms": tau_ms,
        "metrics_rows": int(len(metrics_df)),
        "configurations": int(len(action_ids)),
        "startup_rows": int(len(startup_df)),
        "evaluation_rows": int(len(evaluation_df)),
        "startup_evaluation_overlap": int(overlap_count),
        "startup_trigger_counts": startup_df["trigger_label"].value_counts().to_dict(),
        "evaluation_trigger_counts": evaluation_df["trigger_label"].value_counts().to_dict(),
        "startup_all_rows_violate": bool((startup_df["rt_95"].astype(float) > tau_ms).all()),
        "evaluation_all_rows_violate": bool((evaluation_df["rt_95"].astype(float) > tau_ms).all()),
        "state_definition": "[normalized_rpm, normalized_rt95, 11 current-configuration flags]",
        "observation_size_expected": int(2 + len(FLAG_COLS)),
        "fr_definition": (
            "retained optional abstract functionalities divided by the number "
            "enabled in the current configuration"
        ),
        "fr_zero_current_optional_policy": 1.0,
        "total_optional_feature_groups": int(TOTAL_OPTIONAL_FEATURE_GROUPS),
        "fr_optional_features": list(OPTIONAL_FEATURE_GROUPS.keys()),
        "metrics_p90_rt95_ms": float(metrics_df["rt_95"].astype(float).quantile(0.90)),
    }
    return tau_ms, report


class TransitionAwareEnv(gym.Env if gym is not None else object):
    """One-step contextual DDQN environment for configuration selection."""

    metadata = {"render.modes": []}

    def __init__(
        self,
        metrics_df: pd.DataFrame,
        configs_df: pd.DataFrame,
        startup_df: pd.DataFrame,
        tau_ms: float,
        rt_preprocessor,
        rt_model,
        alpha: float,
        beta: float,
        max_steps: int = 1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0,1], received {alpha}")
        if beta < 0:
            raise ValueError(f"beta must be non-negative, received {beta}")

        self.metrics_df = metrics_df.copy()
        self.configs_df = configs_df.copy()
        self.startup_df = startup_df.copy()
        self.tau_ms = float(tau_ms)
        self.rt_preprocessor = rt_preprocessor
        self.rt_model = rt_model
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_steps = int(max_steps)
        self.eps = float(eps)

        self.action_ids: List[int] = sorted(
            self.configs_df["Config_ID"].astype(int).unique().tolist()
        )
        self.action_index_by_id: Dict[int, int] = {
            config_id: index for index, config_id in enumerate(self.action_ids)
        }

        config_table = (
            self.configs_df[["Config_ID", *FLAG_COLS]]
            .drop_duplicates(subset="Config_ID")
            .copy()
        )
        config_table["Config_ID"] = config_table["Config_ID"].astype(int)
        for col in FLAG_COLS:
            config_table[col] = config_table[col].astype(np.float32)
        config_table = config_table.set_index("Config_ID")

        self.config_flags: Dict[int, np.ndarray] = {
            int(config_id): row.to_numpy(dtype=np.float32)
            for config_id, row in config_table[list(FLAG_COLS)].iterrows()
        }

        self.max_distance = float(len(FLAG_COLS))
        self.rpm_min = float(self.metrics_df["actual_rpm"].astype(float).min())
        self.rpm_max = float(self.metrics_df["actual_rpm"].astype(float).max())
        self.rt_mean = float(self.metrics_df["rt_95"].astype(float).mean())
        self.rt_std = float(self.metrics_df["rt_95"].astype(float).std(ddof=1))

        obs_size = 2 + len(FLAG_COLS)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(obs_size, dtype=np.float32),
            high=np.ones(obs_size, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(len(self.action_ids))

        self.step_count = 0
        self.current_rpm = 0.0
        self.current_rt = 0.0
        self.current_config_id = self.action_ids[0]

    def config_distance(self, from_config: int, to_config: int) -> float:
        from_flags = self.config_flags[int(from_config)]
        to_flags = self.config_flags[int(to_config)]
        return float(np.count_nonzero(from_flags != to_flags))

    def normalized_distance(self, from_config: int, to_config: int) -> float:
        return self.config_distance(from_config, to_config) / self.max_distance

    def optional_feature_presence(self, config_id: int) -> Dict[str, bool]:
        flags = self.config_flags[int(config_id)]
        values = {name: bool(flags[index]) for index, name in enumerate(FLAG_COLS)}
        return {
            feature: any(values[variant] for variant in variants)
            for feature, variants in OPTIONAL_FEATURE_GROUPS.items()
        }

    def functional_retention(
        self,
        current_config: int,
        selected_config: int,
    ) -> Tuple[float, int, int, int]:
        """Return current-relative FR and supporting optional-function counts.

        FR is the number of optional abstract functionalities enabled in both
        the current and selected configurations divided by the number enabled in
        the current configuration. Switching between variants of the same
        abstract functionality preserves that functionality.

        When the current configuration enables no optional functionality, FR is
        defined as 1.0 because the adaptation cannot remove any currently enabled
        optional functionality.
        """
        current = self.optional_feature_presence(current_config)
        selected = self.optional_feature_presence(selected_config)
        retained = sum(int(current[name] and selected[name]) for name in current)
        current_enabled = sum(int(value) for value in current.values())
        selected_enabled = sum(int(value) for value in selected.values())
        fr = 1.0 if current_enabled == 0 else float(retained / current_enabled)
        return fr, int(retained), int(current_enabled), int(selected_enabled)

    def normalized_rt(self, rt: float) -> float:
        return stable_rt_normalization(rt, self.rt_mean, self.rt_std, self.eps)

    def encode_state(self, rpm: float, observed_rt: float, current_config: int) -> np.ndarray:
        norm_rpm = min_max_scale(float(rpm), self.rpm_min, self.rpm_max)
        norm_rt = self.normalized_rt(float(observed_rt))
        current_flags = self.config_flags[int(current_config)]
        return np.concatenate(
            (
                np.asarray([norm_rpm, norm_rt], dtype=np.float32),
                current_flags.astype(np.float32),
            )
        ).astype(np.float32)

    def predict_rt(self, config_id: int, rpm: float) -> float:
        flags = self.config_flags[int(config_id)]
        feature_row = {col: float(flags[index]) for index, col in enumerate(FLAG_COLS)}
        feature_row["actual_rpm"] = float(rpm)
        frame = pd.DataFrame([feature_row])
        transformed = self.rt_preprocessor.transform(frame)
        predicted_log_rt = float(self.rt_model.predict(transformed)[0])
        return float(np.expm1(predicted_log_rt))

    def candidate_rt(
        self,
        current_config: int,
        candidate_config: int,
        observed_rt: float,
        rpm: float,
    ) -> float:
        # A no-op cannot magically repair an observed violation merely because
        # the surrogate prediction is lower than the live measurement.
        if int(candidate_config) == int(current_config):
            return float(observed_rt)
        return self.predict_rt(candidate_config, rpm)

    def reward_for(
        self,
        current_config: int,
        candidate_config: int,
        candidate_rt: float,
    ) -> Tuple[float, float, float, bool]:
        norm_distance = self.normalized_distance(current_config, candidate_config)
        norm_rt = self.normalized_rt(candidate_rt)
        violates_slo = bool(candidate_rt > self.tau_ms)

        reward = -(
            self.alpha * norm_distance
            + (1.0 - self.alpha) * norm_rt
        )
        if violates_slo:
            reward -= self.beta

        return float(reward), float(norm_distance), float(norm_rt), violates_slo

    def reset(self):
        self.step_count = 0
        row = self.startup_df.iloc[np.random.randint(0, len(self.startup_df))]
        self.current_rpm = float(row["actual_rpm"])
        self.current_rt = float(row["rt_95"])
        self.current_config_id = int(row["config"])
        return self.encode_state(
            self.current_rpm,
            self.current_rt,
            self.current_config_id,
        )

    def step(self, action: int):
        self.step_count += 1

        candidate_config = int(self.action_ids[int(action)])
        candidate_rt = self.candidate_rt(
            current_config=self.current_config_id,
            candidate_config=candidate_config,
            observed_rt=self.current_rt,
            rpm=self.current_rpm,
        )
        reward, norm_distance, norm_rt, violates_slo = self.reward_for(
            current_config=self.current_config_id,
            candidate_config=candidate_config,
            candidate_rt=candidate_rt,
        )

        next_observation = self.encode_state(
            self.current_rpm,
            candidate_rt,
            candidate_config,
        )

        terminated = self.step_count >= self.max_steps
        info = {
            "current_config": int(self.current_config_id),
            "selected_config": int(candidate_config),
            "observed_rt": float(self.current_rt),
            "candidate_rt": float(candidate_rt),
            "normalized_rt": float(norm_rt),
            "normalized_distance": float(norm_distance),
            "slo_compliant": bool(not violates_slo),
        }

        self.current_config_id = candidate_config
        self.current_rt = candidate_rt

        return next_observation, reward, terminated, False, info


if nn is not None:
    class DuelingQFunction(nn.Module):
        def __init__(self, observation_size: int, number_of_actions: int, hidden_size: int = 128):
            super().__init__()
            self.fc1 = nn.Linear(observation_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc_advantage = nn.Linear(hidden_size, number_of_actions)
            self.fc_value = nn.Linear(hidden_size, 1)

        def forward(self, x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            if x.ndim == 1:
                x = x.unsqueeze(0)

            hidden = F.relu(self.fc1(x))
            hidden = F.relu(self.fc2(hidden))
            advantages = self.fc_advantage(hidden)
            values = self.fc_value(hidden)
            q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
            return action_value.DiscreteActionValue(q_values)
else:
    class DuelingQFunction:  # pragma: no cover - validation-only fallback
        pass


def detect_convergence(
    episode_rewards: Sequence[float],
    smoothing_window: int = 100,
    plateau_window: int = 500,
    stable_window: int = 200,
) -> int | None:
    if not episode_rewards:
        return None

    smoothed = (
        pd.Series(episode_rewards, dtype=float)
        .rolling(window=smoothing_window, min_periods=1)
        .mean()
        .to_numpy()
    )
    tail_size = min(plateau_window, len(smoothed))
    plateau_mean = float(np.mean(smoothed[-tail_size:]))
    tolerance = max(0.01, 0.10 * abs(plateau_mean))

    for index in range(0, max(0, len(smoothed) - stable_window + 1)):
        window = smoothed[index : index + stable_window]
        if len(window) < stable_window:
            break
        close_fraction = float(np.mean(np.abs(window - plateau_mean) <= tolerance))
        if close_fraction >= 0.90:
            return int(index + smoothing_window - 1)
    return None


def classify_state(row: pd.Series, tau_ms: float) -> str:
    observed_rt = float(row["rt_95"])
    label = str(row.get("trigger_label", ""))

    if observed_rt > tau_ms:
        return "violation"
    if "near" in label.lower():
        return "near_threshold"
    return "safe"


def oracle_rankings(
    env: TransitionAwareEnv,
    current_config: int,
    rpm: float,
    observed_rt: float,
) -> List[Dict[str, float | int | bool]]:
    candidates: List[Dict[str, float | int | bool]] = []

    for config_id in env.action_ids:
        candidate_rt = env.candidate_rt(
            current_config=current_config,
            candidate_config=config_id,
            observed_rt=observed_rt,
            rpm=rpm,
        )
        reward, distance, norm_rt, violates_slo = env.reward_for(
            current_config=current_config,
            candidate_config=config_id,
            candidate_rt=candidate_rt,
        )
        candidates.append(
            {
                "config": int(config_id),
                "reward": float(reward),
                "rt": float(candidate_rt),
                "distance": float(distance),
                "normalized_rt": float(norm_rt),
                "slo_compliant": bool(not violates_slo),
            }
        )

    candidates.sort(
        key=lambda item: (
            -float(item["reward"]),
            float(item["distance"]),
            float(item["rt"]),
            int(item["config"]),
        )
    )
    return candidates


def percentage(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.astype(float).mean() * 100.0)


def summarize_group(frame: pd.DataFrame, group_name: str) -> Dict[str, object]:
    if frame.empty:
        return {
            "group": group_name,
            "states": 0,
        }

    successful = frame[frame["agent_slo_compliant"]].copy()
    result: Dict[str, object] = {
        "group": group_name,
        "states": int(len(frame)),
        "oracle_match_pct": percentage(frame["match"]),
        "mean_regret": float(frame["regret"].mean()),
        "median_regret": float(frame["regret"].median()),
        "regret_le_0.01_pct": float((frame["regret"] <= 0.01).mean() * 100.0),
        "regret_le_0.05_pct": float((frame["regret"] <= 0.05).mean() * 100.0),
        "regret_le_0.10_pct": float((frame["regret"] <= 0.10).mean() * 100.0),
        "agent_slo_compliance_pct": percentage(frame["agent_slo_compliant"]),
        "oracle_slo_compliance_pct": percentage(frame["oracle_slo_compliant"]),
        "mean_agent_rt95_ms": float(frame["agent_rt95_ms"].mean()),
        "median_agent_rt95_ms": float(frame["agent_rt95_ms"].median()),
        "mean_agent_distance": float(frame["agent_distance"].mean()),
        "median_agent_distance": float(frame["agent_distance"].median()),
        "mean_oracle_distance": float(frame["oracle_distance"].mean()),
        "mean_agent_changed_flags": float(frame["agent_changed_flags"].mean()),
        "median_agent_changed_flags": float(frame["agent_changed_flags"].median()),
        "mean_oracle_changed_flags": float(frame["oracle_changed_flags"].mean()),
        "agent_no_change_pct": percentage(frame["agent_no_change"]),
        "oracle_no_change_pct": percentage(frame["oracle_no_change"]),
        "mean_agent_fr": float(frame["agent_fr"].mean()),
        "median_agent_fr": float(frame["agent_fr"].median()),
        "mean_oracle_fr": float(frame["oracle_fr"].mean()),
        "mean_agent_retained_optional_count": float(
            frame["agent_retained_optional_count"].mean()
        ),
        "mean_oracle_retained_optional_count": float(
            frame["oracle_retained_optional_count"].mean()
        ),
        "successful_recovery_states": int(len(successful)),
        "mean_agent_distance_successful": (
            float(successful["agent_distance"].mean()) if len(successful) else float("nan")
        ),
        "mean_agent_changed_flags_successful": (
            float(successful["agent_changed_flags"].mean())
            if len(successful)
            else float("nan")
        ),
        "mean_agent_fr_successful": (
            float(successful["agent_fr"].mean()) if len(successful) else float("nan")
        ),
        "minimum_distance_recovery_pct": (
            percentage(successful["agent_minimum_distance_recovery"])
            if len(successful)
            else float("nan")
        ),
        "mean_excess_distance_successful": (
            float(successful["agent_excess_distance"].mean())
            if len(successful)
            else float("nan")
        ),
        "mean_excess_changed_flags_successful": (
            float(successful["agent_excess_changed_flags"].mean())
            if len(successful)
            else float("nan")
        ),
    }
    return result


def evaluate_policy(
    q_function: DuelingQFunction,
    env: TransitionAwareEnv,
    evaluation_df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    q_function.eval()
    records: List[Dict[str, object]] = []

    with torch.no_grad():
        for row_index, row in evaluation_df.iterrows():
            rpm = float(row["actual_rpm"])
            observed_rt = float(row["rt_95"])
            current_config = int(row["config"])
            observation = env.encode_state(rpm, observed_rt, current_config)

            q_output = q_function(
                torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            )
            q_values = q_output.q_values.detach().cpu().numpy().reshape(-1)

            selected_action_index = int(np.argmax(q_values))
            selected_config = int(env.action_ids[selected_action_index])

            agent_rt = env.candidate_rt(
                current_config=current_config,
                candidate_config=selected_config,
                observed_rt=observed_rt,
                rpm=rpm,
            )
            agent_reward, agent_distance, _, agent_violates = env.reward_for(
                current_config=current_config,
                candidate_config=selected_config,
                candidate_rt=agent_rt,
            )

            oracle_candidates = oracle_rankings(
                env=env,
                current_config=current_config,
                rpm=rpm,
                observed_rt=observed_rt,
            )
            oracle_best = oracle_candidates[0]
            oracle_config = int(oracle_best["config"])
            oracle_reward = float(oracle_best["reward"])

            agent_changed_flags = int(env.config_distance(current_config, selected_config))
            oracle_changed_flags = int(env.config_distance(current_config, oracle_config))

            agent_fr, agent_retained, initial_optional_count, agent_optional_count = (
                env.functional_retention(current_config, selected_config)
            )
            oracle_fr, oracle_retained, _, oracle_optional_count = (
                env.functional_retention(current_config, oracle_config)
            )

            feasible_candidates = [
                candidate for candidate in oracle_candidates if bool(candidate["slo_compliant"])
            ]
            if feasible_candidates:
                minimum_feasible_distance = min(
                    float(candidate["distance"]) for candidate in feasible_candidates
                )
                minimum_feasible_changed_flags = int(
                    round(minimum_feasible_distance * env.max_distance)
                )
            else:
                minimum_feasible_distance = float("nan")
                minimum_feasible_changed_flags = -1

            agent_slo_compliant = bool(not agent_violates)
            if agent_slo_compliant and np.isfinite(minimum_feasible_distance):
                agent_excess_distance = max(
                    0.0, float(agent_distance) - float(minimum_feasible_distance)
                )
                agent_excess_changed_flags = max(
                    0, agent_changed_flags - minimum_feasible_changed_flags
                )
                agent_minimum_distance_recovery = bool(
                    np.isclose(agent_distance, minimum_feasible_distance)
                )
            else:
                agent_excess_distance = float("nan")
                agent_excess_changed_flags = float("nan")
                agent_minimum_distance_recovery = False

            top_agent_indices = np.argsort(q_values)[::-1][:5]
            top_agent_configs = [int(env.action_ids[int(index)]) for index in top_agent_indices]
            top_oracle_configs = [
                int(candidate["config"]) for candidate in oracle_candidates[:5]
            ]

            record = {column: row[column] for column in evaluation_df.columns}
            record.update(
                {
                    "state_group": classify_state(row, env.tau_ms),
                    "agent_config": selected_config,
                    "agent_reward": float(agent_reward),
                    "agent_rt95_ms": float(agent_rt),
                    "agent_distance": float(agent_distance),
                    "agent_changed_flags": int(agent_changed_flags),
                    "agent_slo_compliant": agent_slo_compliant,
                    "agent_no_change": bool(selected_config == current_config),
                    "initial_optional_enabled_count": int(initial_optional_count),
                    "current_optional_enabled_count": int(initial_optional_count),
                    "agent_fr_denominator_count": int(initial_optional_count),
                    "agent_optional_enabled_count": int(agent_optional_count),
                    "agent_retained_optional_count": int(agent_retained),
                    "agent_fr": float(agent_fr),
                    "oracle_config": oracle_config,
                    "oracle_reward": oracle_reward,
                    "oracle_rt95_ms": float(oracle_best["rt"]),
                    "oracle_distance": float(oracle_best["distance"]),
                    "oracle_changed_flags": int(oracle_changed_flags),
                    "oracle_slo_compliant": bool(oracle_best["slo_compliant"]),
                    "oracle_no_change": bool(oracle_config == current_config),
                    "oracle_fr_denominator_count": int(initial_optional_count),
                    "oracle_optional_enabled_count": int(oracle_optional_count),
                    "oracle_retained_optional_count": int(oracle_retained),
                    "oracle_fr": float(oracle_fr),
                    "minimum_feasible_distance": float(minimum_feasible_distance),
                    "minimum_feasible_changed_flags": int(minimum_feasible_changed_flags),
                    "agent_excess_distance": float(agent_excess_distance),
                    "agent_excess_changed_flags": float(agent_excess_changed_flags),
                    "agent_minimum_distance_recovery": bool(
                        agent_minimum_distance_recovery
                    ),
                    "match": bool(selected_config == oracle_config),
                    "regret": float(max(0.0, oracle_reward - agent_reward)),
                    "top5_agent": json.dumps(top_agent_configs),
                    "top5_oracle": json.dumps(top_oracle_configs),
                    "top5_overlap": int(
                        len(set(top_agent_configs).intersection(top_oracle_configs))
                    ),
                }
            )
            records.append(record)

    details_df = pd.DataFrame(records)
    if not (details_df["state_group"] == "violation").all():
        raise RuntimeError("Evaluation produced non-violation states unexpectedly.")
    summary_df = pd.DataFrame([summarize_group(details_df, "violation")])

    details_df.to_csv(output_dir / "evaluation_details.csv", index=False)
    summary_df.to_csv(output_dir / "evaluation_summary.csv", index=False)
    return details_df, summary_df


def plot_training_rewards(rewards: Sequence[float], output_dir: Path) -> None:
    series = pd.Series(rewards, dtype=float)
    moving_average = series.rolling(window=100, min_periods=1).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(moving_average, linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DDQN Training Reward (100-Episode Moving Average)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "training_reward_moving_average.png", dpi=180)
    plt.close()


def train(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=output_dir / "training.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    metrics_df = pd.read_csv(args.metrics)
    configs_df = pd.read_csv(args.configs)
    startup_df = pd.read_csv(args.startup)
    evaluation_df = pd.read_csv(args.evaluation)

    tau_ms, validation_report = validate_inputs(
        metrics_df=metrics_df,
        configs_df=configs_df,
        startup_df=startup_df,
        evaluation_df=evaluation_df,
        explicit_tau=args.tau,
    )

    validation_report.update(
        {
            "alpha": float(args.alpha),
            "beta": float(args.beta),
            "seed": int(args.seed),
            "training_steps": int(args.steps),
        }
    )
    with (output_dir / "input_validation.json").open("w", encoding="utf-8") as file:
        json.dump(validation_report, file, indent=2)

    print(json.dumps(validation_report, indent=2))
    if args.validate_only:
        print("Validation completed successfully; no training was run.")
        return

    if TRAINING_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Training dependencies are missing. Run this script in the existing "
            "JNN/PFRL environment where gym, pfrl, torch, and joblib are installed. "
            f"Original import error: {TRAINING_IMPORT_ERROR}"
        )

    rt_preprocessor = joblib.load(args.rt_preprocessor)
    rt_model = joblib.load(args.rt_model)

    env = TransitionAwareEnv(
        metrics_df=metrics_df,
        configs_df=configs_df,
        startup_df=startup_df,
        tau_ms=tau_ms,
        rt_preprocessor=rt_preprocessor,
        rt_model=rt_model,
        alpha=args.alpha,
        beta=args.beta,
        max_steps=1,
    )
    env.action_space.seed(args.seed)

    observation_size = int(env.observation_space.shape[0])
    number_of_actions = int(env.action_space.n)
    q_function = DuelingQFunction(
        observation_size=observation_size,
        number_of_actions=number_of_actions,
        hidden_size=args.hidden_size,
    )

    optimizer = torch.optim.Adam(q_function.parameters(), lr=args.learning_rate)
    explorer = explorers.LinearDecayEpsilonGreedy(
        start_epsilon=1.0,
        end_epsilon=0.05,
        decay_steps=args.epsilon_decay_steps,
        random_action_func=lambda: env.action_space.sample(),
    )
    replay_buffer = replay_buffers.ReplayBuffer(capacity=args.replay_capacity)

    agent = agents.DoubleDQN(
        q_function=q_function,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        gamma=args.gamma,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        gpu=-1,
        target_update_interval=args.target_update_interval,
    )

    start_time = time.time()
    observation = env.reset()
    episode_rewards: List[float] = []

    for step in range(args.steps):
        action = agent.act(observation)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        agent.observe(next_observation, reward, done, reset=done)

        if done:
            episode_rewards.append(float(reward))
            observation = env.reset()
        else:
            observation = next_observation

    elapsed_seconds = time.time() - start_time
    convergence_episode = detect_convergence(episode_rewards)

    training_df = pd.DataFrame(
        {
            "episode": np.arange(1, len(episode_rewards) + 1),
            "reward": episode_rewards,
        }
    )
    training_df.to_csv(output_dir / "training_rewards.csv", index=False)
    plot_training_rewards(episode_rewards, output_dir)

    torch.save(
        {
            "model_state_dict": q_function.state_dict(),
            "observation_size": observation_size,
            "number_of_actions": number_of_actions,
            "flag_columns": list(FLAG_COLS),
            "optional_feature_groups": {
                name: list(variants) for name, variants in OPTIONAL_FEATURE_GROUPS.items()
            },
            "fr_definition": (
                "retained optional abstract functionalities / optional abstract "
                "functionalities enabled in the current configuration"
            ),
            "fr_zero_current_optional_policy": 1.0,
            "total_optional_feature_groups": int(TOTAL_OPTIONAL_FEATURE_GROUPS),
            "alpha": float(args.alpha),
            "beta": float(args.beta),
            "tau_ms": float(tau_ms),
            "seed": int(args.seed),
        },
        output_dir / "ddqn_model.pt",
    )

    details_df, summary_df = evaluate_policy(
        q_function=q_function,
        env=env,
        evaluation_df=evaluation_df,
        output_dir=output_dir,
    )

    run_summary = {
        **validation_report,
        "observation_size": observation_size,
        "number_of_actions": number_of_actions,
        "training_time_seconds": float(elapsed_seconds),
        "steps_per_second": float(args.steps / elapsed_seconds),
        "mean_training_reward": float(np.mean(episode_rewards)),
        "std_training_reward": float(np.std(episode_rewards)),
        "approx_convergence_episode": convergence_episode,
        "overall_evaluation": summary_df.iloc[0].to_dict(),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as file:
        json.dump(run_summary, file, indent=2, default=str)

    print("\nTraining completed.")
    print(f"Time: {elapsed_seconds:.2f} seconds")
    print(f"Steps/second: {args.steps / elapsed_seconds:.2f}")
    print(f"Approximate convergence episode: {convergence_episode}")
    print("\nEvaluation summary:")
    print(summary_df.to_string(index=False))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Violation-only transition-aware DDQN with FR metrics for JNN tail-latency recovery."
    )
    parser.add_argument(
        "--metrics",
        default="Data/DataConf/Final_DS_JNN9_local_Desktop_UpdatedWed_all.csv",
    )
    parser.add_argument(
        "--configs",
        default="Data/DataConf/ALL_Config_binary.csv",
    )
    parser.add_argument(
        "--startup",
        default="Data/DataConf/JNN_startup_violation_only.csv",
    )
    parser.add_argument(
        "--evaluation",
        default="Data/DataConf/JNN_evaluation_violation_only.csv",
    )
    parser.add_argument(
        "--rt-preprocessor",
        default="PM_Models_Conf/Models/JNN_RT_preproc_binary_rt95.pkl",
    )
    parser.add_argument(
        "--rt-model",
        default="PM_Models_Conf/Models/JNN_RT_XGB_binary_rt95.pkl",
    )
    parser.add_argument("--output-dir", default="Results_transition_aware")
    parser.add_argument("--alpha", type=float, default=0.90)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--steps", type=int, default=120000)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.90)
    parser.add_argument("--epsilon-decay-steps", type=int, default=50000)
    parser.add_argument("--replay-capacity", type=int, default=100000)
    parser.add_argument("--replay-start-size", type=int, default=1000)
    parser.add_argument("--target-update-interval", type=int, default=2000)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate datasets/configurations without loading PPM models or training.",
    )
    return parser


if __name__ == "__main__":
    train(build_argument_parser().parse_args())
