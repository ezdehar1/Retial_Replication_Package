#!/usr/bin/env python3
"""
Transition-aware DAGAME-style GA baseline for JNN tail-latency recovery.

Mission of this baseline
------------------------
Evaluate evolutionary runtime search under exactly the same transition-aware
objective used by the proposed DDQN:

    reward = -[alpha * normalized_hamming_distance
               + (1-alpha) * normalized_rt95]
             - beta * I(candidate_rt95 > tau)

The experiment is violation-only: every evaluation row must satisfy
observed RT95 > tau. For each state, the GA searches among DSPL-valid JNN
configurations. It uses the same PPM, RT normalization, no-op handling,
architectural-distance definition, and current-relative Functional Retention
(FR) definition as the transition-aware DDQN experiment.

The evolutionary mechanics remain DAGAME-inspired:
  * binary feature-vector chromosomes;
  * nearest-valid-configuration repair by Hamming distance;
  * random-mask crossover;
  * one-gene mutation;
  * replacement of the least-favourable population member.

Typical command
---------------
python DAGAME_GA_transition_aware_violation.py \
  --alpha 0.90 --beta 3.0 --seed 45 --pop 20 --gens 10 \
  --metrics Data/DataConf/Final_DS_JNN9_local_Desktop_UpdatedWed_all.csv \
  --configs Data/DataConf/ALL_Config_binary.csv \
  --evaluation Data/DataConf/JNN_evaluation_violation_only.csv \
  --rt-preprocessor PM_Models_Conf/Models/JNN_RT_preproc_binary_rt95.pkl \
  --rt-model PM_Models_Conf/Models/JNN_RT_XGB_binary_rt95.pkl \
  --ddqn-details Results_violation/alpha_0.90_seed_45/evaluation_details.csv \
  --output-dir Results_violation/ga_alpha_0.90_seed_45
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

# FR is measured relative to the optional abstract functionalities enabled in
# the current configuration. Alternative implementations of one functionality
# count as the same retained functionality.
OPTIONAL_FEATURE_GROUPS: Mapping[str, Tuple[str, ...]] = {
    "media": ("media1", "media2"),
    "analytics": ("analytics1", "analytics2"),
    "recommendation": ("recommendation1", "recommendation2"),
    "breaking": ("breaking",),
    "advertisement": ("adv1", "adv2"),
}

EVALUATION_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "config",
    "actual_rpm",
    "rt_95",
    "tau_ms",
)

ALIGNMENT_KEYS: Tuple[str, ...] = (
    "run_time_s",
    "timestamp",
    "config",
    "actual_rpm",
    "rt_95",
)


@dataclass(frozen=True)
class CandidateEvaluation:
    config_id: int
    reward: float
    rt95_ms: float
    normalized_rt: float
    distance: float
    changed_flags: int
    slo_compliant: bool

    @property
    def rank_key(self) -> Tuple[float, float, float, int]:
        """Larger tuple is better; tie-breaking matches the DDQN oracle."""
        return (
            float(self.reward),
            -float(self.distance),
            -float(self.rt95_ms),
            -int(self.config_id),
        )


@dataclass
class GASearchResult:
    selected: CandidateEvaluation
    decision_latency_ms: float
    unique_evaluated_configs: int
    mean_repair_hamming: float
    max_repair_hamming: int
    evaluated_config_ids: List[int]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stable_rt_normalization(
    rt: float,
    mean: float,
    std: float,
    eps: float = 1e-8,
) -> float:
    """Map an RT z-score smoothly into [0, 1], matching the DDQN code."""
    z_score = (float(rt) - float(mean)) / (float(std) + float(eps))
    return float((np.tanh(z_score) + 1.0) / 2.0)


def percentage(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.astype(float).mean() * 100.0)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def validate_inputs(
    metrics_df: pd.DataFrame,
    configs_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    explicit_tau: Optional[float],
) -> Tuple[float, Dict[str, Any]]:
    missing_metrics = [
        column for column in ("actual_rpm", "rt_95") if column not in metrics_df.columns
    ]
    if missing_metrics:
        raise ValueError(f"Metrics CSV is missing columns: {missing_metrics}")

    missing_configs = [
        column for column in ("Config_ID", *FLAG_COLS) if column not in configs_df.columns
    ]
    if missing_configs:
        raise ValueError(f"Configuration CSV is missing columns: {missing_configs}")

    missing_evaluation = [
        column for column in EVALUATION_REQUIRED_COLUMNS if column not in evaluation_df.columns
    ]
    if missing_evaluation:
        raise ValueError(f"Evaluation CSV is missing columns: {missing_evaluation}")
    if evaluation_df.empty:
        raise ValueError("Evaluation CSV is empty.")

    duplicated_ids = configs_df["Config_ID"].astype(int).duplicated().sum()
    if duplicated_ids:
        raise ValueError(f"Configuration CSV contains {duplicated_ids} duplicated Config_ID rows.")

    for column in FLAG_COLS:
        values = set(configs_df[column].dropna().astype(int).unique().tolist())
        if not values.issubset({0, 1}):
            raise ValueError(f"Configuration flag {column!r} is not binary: {sorted(values)}")

    available_ids = set(configs_df["Config_ID"].astype(int).tolist())
    missing_current_ids = sorted(
        set(evaluation_df["config"].astype(int).tolist()).difference(available_ids)
    )
    if missing_current_ids:
        raise ValueError(
            "Evaluation states reference unknown configurations: "
            f"{missing_current_ids[:10]}"
        )

    tau_values = sorted(evaluation_df["tau_ms"].astype(float).unique().tolist())
    if len(tau_values) != 1:
        raise ValueError(f"Evaluation CSV contains multiple tau values: {tau_values}")
    dataset_tau = float(tau_values[0])
    tau_ms = float(explicit_tau) if explicit_tau is not None else dataset_tau

    if explicit_tau is not None and not np.isclose(tau_ms, dataset_tau):
        logging.warning(
            "Explicit tau %.4f differs from evaluation-dataset tau %.4f.",
            tau_ms,
            dataset_tau,
        )

    nonviolations = int((evaluation_df["rt_95"].astype(float) <= tau_ms).sum())
    if nonviolations:
        raise ValueError(
            "This GA experiment is violation-only, but the evaluation CSV contains "
            f"{nonviolations} compliant states."
        )

    report: Dict[str, Any] = {
        "tau_ms": tau_ms,
        "metrics_rows": int(len(metrics_df)),
        "evaluation_rows": int(len(evaluation_df)),
        "configurations": int(len(configs_df)),
        "all_evaluation_rows_violate": bool(
            (evaluation_df["rt_95"].astype(float) > tau_ms).all()
        ),
        "metrics_rt95_mean": float(metrics_df["rt_95"].astype(float).mean()),
        "metrics_rt95_std_sample": float(
            metrics_df["rt_95"].astype(float).std(ddof=1)
        ),
        "metrics_p90_rt95_ms": float(
            metrics_df["rt_95"].astype(float).quantile(0.90)
        ),
        "reward": "-[alpha*distance + (1-alpha)*normalized_rt95] - beta*I[rt95>tau]",
        "alpha_default_for_proposed_comparison": 0.90,
        "fr_definition": (
            "retained optional functionalities / optional functionalities "
            "enabled in the current configuration"
        ),
        "fr_denominator_mode": "current_optional_enabled_count",
        "fr_zero_denominator_rule": "FR=1.0",
        "fr_optional_features": list(OPTIONAL_FEATURE_GROUPS.keys()),
        "no_op_rule": "current configuration retains the observed RT95",
    }
    return tau_ms, report


class TransitionAwareProblem:
    """Shared transition-aware objective used by GA and exhaustive oracle."""

    def __init__(
        self,
        metrics_df: pd.DataFrame,
        configs_df: pd.DataFrame,
        tau_ms: float,
        rt_preprocessor: Any,
        rt_model: Any,
        alpha: float,
        beta: float,
        eps: float = 1e-8,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0,1], received {alpha}")
        if beta < 0.0:
            raise ValueError(f"beta must be non-negative, received {beta}")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau_ms = float(tau_ms)
        self.rt_preprocessor = rt_preprocessor
        self.rt_model = rt_model
        self.eps = float(eps)

        table = (
            configs_df[["Config_ID", *FLAG_COLS]]
            .drop_duplicates(subset="Config_ID")
            .copy()
            .sort_values("Config_ID")
            .reset_index(drop=True)
        )
        table["Config_ID"] = table["Config_ID"].astype(int)
        for column in FLAG_COLS:
            table[column] = table[column].astype(int)

        self.configs = table
        self.valid_X = table[list(FLAG_COLS)].to_numpy(dtype=int)
        self.config_ids = table["Config_ID"].to_numpy(dtype=int)
        self.max_distance = float(len(FLAG_COLS))
        self.n_genes = len(FLAG_COLS)

        self.id_to_index: Dict[int, int] = {
            int(config_id): int(index)
            for index, config_id in enumerate(self.config_ids)
        }
        self.tuple_to_index: Dict[Tuple[int, ...], int] = {}
        for index, vector in enumerate(self.valid_X):
            self.tuple_to_index.setdefault(tuple(vector.tolist()), int(index))

        self.rt_mean = float(metrics_df["rt_95"].astype(float).mean())
        self.rt_std = float(metrics_df["rt_95"].astype(float).std(ddof=1))

    def flags_for(self, config_id: int) -> np.ndarray:
        try:
            index = self.id_to_index[int(config_id)]
        except KeyError as exc:
            raise ValueError(f"Unknown Config_ID: {config_id}") from exc
        return self.valid_X[index].copy()

    def config_distance(self, from_config: int, to_config: int) -> int:
        from_flags = self.flags_for(from_config)
        to_flags = self.flags_for(to_config)
        return int(np.count_nonzero(from_flags != to_flags))

    def normalized_distance(self, from_config: int, to_config: int) -> float:
        return float(self.config_distance(from_config, to_config) / self.max_distance)

    def optional_feature_presence(self, config_id: int) -> Dict[str, bool]:
        vector = self.flags_for(config_id)
        values = {name: bool(vector[index]) for index, name in enumerate(FLAG_COLS)}
        return {
            feature: any(values[variant] for variant in variants)
            for feature, variants in OPTIONAL_FEATURE_GROUPS.items()
        }

    def functional_retention(
        self,
        initial_config: int,
        selected_config: int,
    ) -> Tuple[float, int, int, int]:
        """Return current-relative FR and supporting optional-function counts.

        FR is the fraction of optional abstract functionalities enabled in the
        current configuration that remain enabled in the selected configuration.
        Switching between variants of the same abstract functionality preserves
        that functionality. When the current configuration enables no optional
        functionality, FR is defined as 1.0 because nothing can be lost.
        """
        initial = self.optional_feature_presence(initial_config)
        selected = self.optional_feature_presence(selected_config)
        retained = sum(int(initial[name] and selected[name]) for name in initial)
        initial_enabled = sum(int(value) for value in initial.values())
        selected_enabled = sum(int(value) for value in selected.values())
        fr = 1.0 if initial_enabled == 0 else float(retained / initial_enabled)
        return (
            float(fr),
            int(retained),
            int(initial_enabled),
            int(selected_enabled),
        )

    def normalized_rt(self, rt95_ms: float) -> float:
        return stable_rt_normalization(rt95_ms, self.rt_mean, self.rt_std, self.eps)

    def predict_rt(self, config_id: int, rpm: float) -> float:
        vector = self.flags_for(config_id)
        feature_row = {
            column: float(vector[index]) for index, column in enumerate(FLAG_COLS)
        }
        feature_row["actual_rpm"] = float(rpm)
        frame = pd.DataFrame([feature_row], columns=[*FLAG_COLS, "actual_rpm"])
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
        # Exact match with the DDQN experiment: a no-op cannot be credited with
        # a lower PPM prediction when the currently observed state violates tau.
        if int(candidate_config) == int(current_config):
            return float(observed_rt)
        return self.predict_rt(candidate_config, rpm)

    def evaluate_candidate(
        self,
        current_config: int,
        candidate_config: int,
        observed_rt: float,
        rpm: float,
    ) -> CandidateEvaluation:
        candidate_rt = self.candidate_rt(
            current_config=current_config,
            candidate_config=candidate_config,
            observed_rt=observed_rt,
            rpm=rpm,
        )
        changed_flags = self.config_distance(current_config, candidate_config)
        distance = float(changed_flags / self.max_distance)
        normalized_rt = self.normalized_rt(candidate_rt)
        violates_slo = bool(candidate_rt > self.tau_ms)

        reward = -(
            self.alpha * distance
            + (1.0 - self.alpha) * normalized_rt
        )
        if violates_slo:
            reward -= self.beta

        return CandidateEvaluation(
            config_id=int(candidate_config),
            reward=float(reward),
            rt95_ms=float(candidate_rt),
            normalized_rt=float(normalized_rt),
            distance=float(distance),
            changed_flags=int(changed_flags),
            slo_compliant=bool(not violates_slo),
        )

    def transform_to_valid(
        self,
        chromosome: np.ndarray,
        rng: random.Random,
    ) -> Tuple[np.ndarray, int, int]:
        chromosome = np.asarray(chromosome, dtype=int)
        key = tuple(chromosome.tolist())
        if key in self.tuple_to_index:
            index = int(self.tuple_to_index[key])
            return self.valid_X[index].copy(), index, 0

        distances = np.sum(self.valid_X != chromosome, axis=1)
        minimum = int(distances.min())
        candidate_indices = np.flatnonzero(distances == minimum).tolist()
        chosen_index = int(rng.choice(candidate_indices))
        return self.valid_X[chosen_index].copy(), chosen_index, minimum

    def oracle_rankings(
        self,
        current_config: int,
        observed_rt: float,
        rpm: float,
    ) -> List[CandidateEvaluation]:
        candidates = [
            self.evaluate_candidate(
                current_config=current_config,
                candidate_config=int(config_id),
                observed_rt=observed_rt,
                rpm=rpm,
            )
            for config_id in self.config_ids
        ]
        candidates.sort(key=lambda item: item.rank_key, reverse=True)
        return candidates


class DAGAMEGA:
    """DAGAME-inspired evolutionary search over DSPL feature vectors."""

    def __init__(
        self,
        problem: TransitionAwareProblem,
        population_size: int,
        generations: int,
        crossover_probability: float,
        mutation_probability: float,
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if generations < 0:
            raise ValueError("generations must be non-negative.")
        if not 0.0 <= crossover_probability <= 1.0:
            raise ValueError("crossover_probability must be in [0,1].")
        if not 0.0 <= mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be in [0,1].")

        self.problem = problem
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.crossover_probability = float(crossover_probability)
        self.mutation_probability = float(mutation_probability)

    def choose_config(
        self,
        current_config: int,
        observed_rt: float,
        rpm: float,
        seed: int,
    ) -> GASearchResult:
        rng = random.Random(int(seed))
        np_rng = np.random.default_rng(int(seed))

        evaluation_cache: Dict[int, CandidateEvaluation] = {}
        repair_distances: List[int] = []

        def evaluate(chromosome: np.ndarray) -> CandidateEvaluation:
            valid_vector, valid_index, _ = self.problem.transform_to_valid(chromosome, rng)
            config_id = int(self.problem.config_ids[valid_index])
            if config_id not in evaluation_cache:
                evaluation_cache[config_id] = self.problem.evaluate_candidate(
                    current_config=current_config,
                    candidate_config=config_id,
                    observed_rt=observed_rt,
                    rpm=rpm,
                )
            return evaluation_cache[config_id]

        def score(chromosome: np.ndarray) -> Tuple[float, float, float, int]:
            return evaluate(chromosome).rank_key

        start = perf_counter()

        population: List[np.ndarray] = []
        for _ in range(self.population_size):
            raw = np_rng.integers(
                0,
                2,
                size=self.problem.n_genes,
                dtype=int,
            )
            valid, _, repair_distance = self.problem.transform_to_valid(raw, rng)
            population.append(valid)
            repair_distances.append(int(repair_distance))

        best_chromosome = max(population, key=score).copy()
        best_evaluation = evaluate(best_chromosome)

        for _ in range(self.generations):
            ranked = sorted(population, key=score, reverse=True)
            parent1 = ranked[0].copy()
            parent2 = ranked[1].copy()

            if rng.random() < self.crossover_probability:
                mask = np_rng.integers(
                    0,
                    2,
                    size=self.problem.n_genes,
                    dtype=int,
                )
                child = np.where(mask == 0, parent1, parent2).astype(int)
            else:
                child = parent1.copy()

            if rng.random() < self.mutation_probability:
                mutation_index = rng.randrange(self.problem.n_genes)
                child[mutation_index] = 1 - child[mutation_index]

            child, _, repair_distance = self.problem.transform_to_valid(child, rng)
            repair_distances.append(int(repair_distance))
            child_evaluation = evaluate(child)

            worst_position = min(
                range(len(population)),
                key=lambda index: score(population[index]),
            )
            if child_evaluation.rank_key > evaluate(population[worst_position]).rank_key:
                population[worst_position] = child

            if child_evaluation.rank_key > best_evaluation.rank_key:
                best_chromosome = child.copy()
                best_evaluation = child_evaluation

        # Re-evaluate through cache only; this adds no PPM call.
        selected = evaluate(best_chromosome)
        latency_ms = float((perf_counter() - start) * 1000.0)

        return GASearchResult(
            selected=selected,
            decision_latency_ms=latency_ms,
            unique_evaluated_configs=int(len(evaluation_cache)),
            mean_repair_hamming=(
                float(np.mean(repair_distances)) if repair_distances else 0.0
            ),
            max_repair_hamming=(
                int(np.max(repair_distances)) if repair_distances else 0
            ),
            evaluated_config_ids=sorted(int(value) for value in evaluation_cache),
        )


def summarize_ga(details: pd.DataFrame) -> Dict[str, Any]:
    successful = details[details["ga_slo_compliant"]].copy()
    return {
        "method": "DAGAME-GA",
        "states": int(len(details)),
        "alpha": float(details["alpha"].iloc[0]),
        "beta": float(details["beta"].iloc[0]),
        "population_size": int(details["ga_population_size"].iloc[0]),
        "generations": int(details["ga_generations"].iloc[0]),
        "predicted_slo_recovery_pct": percentage(details["ga_slo_compliant"]),
        "successful_recovery_states": int(len(successful)),
        "oracle_match_pct": percentage(details["ga_oracle_match"]),
        "mean_regret": float(details["ga_regret"].mean()),
        "median_regret": float(details["ga_regret"].median()),
        "regret_le_0.01_pct": float((details["ga_regret"] <= 0.01).mean() * 100.0),
        "regret_le_0.05_pct": float((details["ga_regret"] <= 0.05).mean() * 100.0),
        "regret_le_0.10_pct": float((details["ga_regret"] <= 0.10).mean() * 100.0),
        "mean_ga_rt95_ms": float(details["ga_rt95_ms"].mean()),
        "median_ga_rt95_ms": float(details["ga_rt95_ms"].median()),
        "mean_ga_distance": float(details["ga_distance"].mean()),
        "median_ga_distance": float(details["ga_distance"].median()),
        "mean_ga_changed_flags": float(details["ga_changed_flags"].mean()),
        "median_ga_changed_flags": float(details["ga_changed_flags"].median()),
        "ga_no_change_pct": percentage(details["ga_no_change"]),
        "mean_ga_fr": float(details["ga_fr"].mean()),
        "median_ga_fr": float(details["ga_fr"].median()),
        "fr_denominator_mode": "current_optional_enabled_count",
        "mean_ga_retained_optional_count": float(
            details["ga_retained_optional_count"].mean()
        ),
        "mean_ga_distance_successful": (
            float(successful["ga_distance"].mean()) if len(successful) else float("nan")
        ),
        "mean_ga_changed_flags_successful": (
            float(successful["ga_changed_flags"].mean())
            if len(successful)
            else float("nan")
        ),
        "mean_ga_fr_successful": (
            float(successful["ga_fr"].mean()) if len(successful) else float("nan")
        ),
        "minimum_distance_recovery_pct": (
            percentage(successful["ga_minimum_distance_recovery"])
            if len(successful)
            else float("nan")
        ),
        "mean_excess_distance_successful": (
            float(successful["ga_excess_distance"].mean())
            if len(successful)
            else float("nan")
        ),
        "mean_excess_changed_flags_successful": (
            float(successful["ga_excess_changed_flags"].mean())
            if len(successful)
            else float("nan")
        ),
        "mean_decision_latency_ms": float(details["ga_decision_latency_ms"].mean()),
        "median_decision_latency_ms": float(
            details["ga_decision_latency_ms"].median()
        ),
        "p95_decision_latency_ms": float(
            details["ga_decision_latency_ms"].quantile(0.95)
        ),
        "mean_unique_evaluated_configs": float(
            details["ga_unique_evaluated_configs"].mean()
        ),
        "mean_repair_hamming": float(details["ga_mean_repair_hamming"].mean()),
        "maximum_repair_hamming": int(details["ga_max_repair_hamming"].max()),
        "mean_oracle_distance": float(details["oracle_distance"].mean()),
        "mean_oracle_changed_flags": float(details["oracle_changed_flags"].mean()),
        "mean_oracle_fr": float(details["oracle_fr"].mean()),
    }


def evaluate_ga(
    problem: TransitionAwareProblem,
    ga: DAGAMEGA,
    evaluation_df: pd.DataFrame,
    base_seed: int,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records: List[Dict[str, Any]] = []

    for position, (_, row) in enumerate(evaluation_df.iterrows()):
        current_config = int(row["config"])
        rpm = float(row["actual_rpm"])
        observed_rt = float(row["rt_95"])

        state_seed = int(base_seed + position)
        search_result = ga.choose_config(
            current_config=current_config,
            observed_rt=observed_rt,
            rpm=rpm,
            seed=state_seed,
        )
        selected = search_result.selected

        oracle_candidates = problem.oracle_rankings(
            current_config=current_config,
            observed_rt=observed_rt,
            rpm=rpm,
        )
        oracle = oracle_candidates[0]
        feasible_candidates = [
            candidate for candidate in oracle_candidates if candidate.slo_compliant
        ]

        if feasible_candidates:
            minimum_feasible_distance = min(
                candidate.distance for candidate in feasible_candidates
            )
            minimum_feasible_changed_flags = int(
                round(minimum_feasible_distance * problem.max_distance)
            )
        else:
            minimum_feasible_distance = float("nan")
            minimum_feasible_changed_flags = -1

        ga_fr, ga_retained, initial_optional, ga_optional = problem.functional_retention(
            current_config,
            selected.config_id,
        )
        oracle_fr, oracle_retained, _, oracle_optional = problem.functional_retention(
            current_config,
            oracle.config_id,
        )

        if selected.slo_compliant and np.isfinite(minimum_feasible_distance):
            ga_excess_distance = max(
                0.0,
                float(selected.distance - minimum_feasible_distance),
            )
            ga_excess_changed_flags = max(
                0,
                int(selected.changed_flags - minimum_feasible_changed_flags),
            )
            minimum_distance_recovery = bool(
                np.isclose(selected.distance, minimum_feasible_distance)
            )
        else:
            ga_excess_distance = float("nan")
            ga_excess_changed_flags = float("nan")
            minimum_distance_recovery = False

        record = {column: row[column] for column in evaluation_df.columns}
        record.update(
            {
                "alpha": float(problem.alpha),
                "beta": float(problem.beta),
                "ga_seed": state_seed,
                "ga_population_size": int(ga.population_size),
                "ga_generations": int(ga.generations),
                "ga_config": int(selected.config_id),
                "ga_reward": float(selected.reward),
                "ga_rt95_ms": float(selected.rt95_ms),
                "ga_normalized_rt": float(selected.normalized_rt),
                "ga_distance": float(selected.distance),
                "ga_changed_flags": int(selected.changed_flags),
                "ga_slo_compliant": bool(selected.slo_compliant),
                "ga_no_change": bool(selected.config_id == current_config),
                "initial_optional_enabled_count": int(initial_optional),
                "ga_optional_enabled_count": int(ga_optional),
                "ga_retained_optional_count": int(ga_retained),
                "ga_fr": float(ga_fr),
                "ga_decision_latency_ms": float(search_result.decision_latency_ms),
                "ga_unique_evaluated_configs": int(
                    search_result.unique_evaluated_configs
                ),
                "ga_mean_repair_hamming": float(
                    search_result.mean_repair_hamming
                ),
                "ga_max_repair_hamming": int(search_result.max_repair_hamming),
                "ga_evaluated_config_ids": json.dumps(
                    search_result.evaluated_config_ids
                ),
                "oracle_config": int(oracle.config_id),
                "oracle_reward": float(oracle.reward),
                "oracle_rt95_ms": float(oracle.rt95_ms),
                "oracle_distance": float(oracle.distance),
                "oracle_changed_flags": int(oracle.changed_flags),
                "oracle_slo_compliant": bool(oracle.slo_compliant),
                "oracle_no_change": bool(oracle.config_id == current_config),
                "oracle_optional_enabled_count": int(oracle_optional),
                "oracle_retained_optional_count": int(oracle_retained),
                "oracle_fr": float(oracle_fr),
                "minimum_feasible_distance": float(minimum_feasible_distance),
                "minimum_feasible_changed_flags": int(
                    minimum_feasible_changed_flags
                ),
                "ga_excess_distance": float(ga_excess_distance),
                "ga_excess_changed_flags": float(ga_excess_changed_flags),
                "ga_minimum_distance_recovery": bool(minimum_distance_recovery),
                "ga_oracle_match": bool(selected.config_id == oracle.config_id),
                "ga_regret": float(max(0.0, oracle.reward - selected.reward)),
                "top5_oracle": json.dumps(
                    [int(candidate.config_id) for candidate in oracle_candidates[:5]]
                ),
            }
        )
        records.append(record)

        print(
            f"[{position + 1:02d}/{len(evaluation_df)}] "
            f"current={current_config} ga={selected.config_id} "
            f"rt95={selected.rt95_ms:.3f}ms "
            f"changed={selected.changed_flags} "
            f"recovered={selected.slo_compliant} "
            f"latency={search_result.decision_latency_ms:.3f}ms"
        )

    details = pd.DataFrame(records)
    summary = pd.DataFrame([summarize_ga(details)])
    details.to_csv(output_dir / "ga_evaluation_details.csv", index=False)
    summary.to_csv(output_dir / "ga_evaluation_summary.csv", index=False)
    return details, summary


def summarize_ddqn(details: pd.DataFrame, alpha: float) -> Dict[str, Any]:
    successful = details[details["agent_slo_compliant"]].copy()
    result: Dict[str, Any] = {
        "method": f"DDQN alpha={alpha:.2f}",
        "states": int(len(details)),
        "predicted_slo_recovery_pct": percentage(details["agent_slo_compliant"]),
        "mean_rt95_ms": float(details["agent_rt95_ms"].mean()),
        "median_rt95_ms": float(details["agent_rt95_ms"].median()),
        "mean_distance": float(details["agent_distance"].mean()),
        "median_distance": float(details["agent_distance"].median()),
        "mean_changed_flags": float(details["agent_changed_flags"].mean()),
        "median_changed_flags": float(details["agent_changed_flags"].median()),
        "mean_fr": float(details["agent_fr"].mean()),
        "median_fr": float(details["agent_fr"].median()),
        "minimum_distance_recovery_pct": (
            percentage(successful["agent_minimum_distance_recovery"])
            if len(successful)
            else float("nan")
        ),
        "mean_excess_changed_flags_successful": (
            float(successful["agent_excess_changed_flags"].mean())
            if len(successful)
            else float("nan")
        ),
        "oracle_match_pct": percentage(details["match"]),
        "mean_regret": float(details["regret"].mean()),
        "median_regret": float(details["regret"].median()),
        "mean_decision_latency_ms": (
            float(details["agent_decision_latency_ms"].mean())
            if "agent_decision_latency_ms" in details.columns
            else float("nan")
        ),
    }
    return result


def create_comparison(
    ga_details: pd.DataFrame,
    ddqn_details_path: Optional[Path],
    ddqn_alpha: float,
    output_dir: Path,
) -> Optional[pd.DataFrame]:
    ga_successful = ga_details[ga_details["ga_slo_compliant"]]
    rows: List[Dict[str, Any]] = [
        {
            "method": f"DAGAME-GA alpha={ga_details['alpha'].iloc[0]:.2f}",
            "states": int(len(ga_details)),
            "predicted_slo_recovery_pct": percentage(
                ga_details["ga_slo_compliant"]
            ),
            "mean_rt95_ms": float(ga_details["ga_rt95_ms"].mean()),
            "median_rt95_ms": float(ga_details["ga_rt95_ms"].median()),
            "mean_distance": float(ga_details["ga_distance"].mean()),
            "median_distance": float(ga_details["ga_distance"].median()),
            "mean_changed_flags": float(ga_details["ga_changed_flags"].mean()),
            "median_changed_flags": float(
                ga_details["ga_changed_flags"].median()
            ),
            "mean_fr": float(ga_details["ga_fr"].mean()),
            "median_fr": float(ga_details["ga_fr"].median()),
            "minimum_distance_recovery_pct": (
                percentage(ga_successful["ga_minimum_distance_recovery"])
                if len(ga_successful)
                else float("nan")
            ),
            "mean_excess_changed_flags_successful": (
                float(ga_successful["ga_excess_changed_flags"].mean())
                if len(ga_successful)
                else float("nan")
            ),
            "oracle_match_pct": percentage(ga_details["ga_oracle_match"]),
            "mean_regret": float(ga_details["ga_regret"].mean()),
            "median_regret": float(ga_details["ga_regret"].median()),
            "mean_decision_latency_ms": float(
                ga_details["ga_decision_latency_ms"].mean()
            ),
        },
        {
            "method": "Exhaustive oracle",
            "states": int(len(ga_details)),
            "predicted_slo_recovery_pct": percentage(
                ga_details["oracle_slo_compliant"]
            ),
            "mean_rt95_ms": float(ga_details["oracle_rt95_ms"].mean()),
            "median_rt95_ms": float(ga_details["oracle_rt95_ms"].median()),
            "mean_distance": float(ga_details["oracle_distance"].mean()),
            "median_distance": float(ga_details["oracle_distance"].median()),
            "mean_changed_flags": float(
                ga_details["oracle_changed_flags"].mean()
            ),
            "median_changed_flags": float(
                ga_details["oracle_changed_flags"].median()
            ),
            "mean_fr": float(ga_details["oracle_fr"].mean()),
            "median_fr": float(ga_details["oracle_fr"].median()),
            "minimum_distance_recovery_pct": 100.0,
            "mean_excess_changed_flags_successful": 0.0,
            "oracle_match_pct": 100.0,
            "mean_regret": 0.0,
            "median_regret": 0.0,
            "mean_decision_latency_ms": float("nan"),
        },
    ]

    ddqn_details: Optional[pd.DataFrame] = None
    if ddqn_details_path is not None:
        ddqn_details = pd.read_csv(ddqn_details_path)
        required = {
            "agent_config",
            "agent_rt95_ms",
            "agent_distance",
            "agent_changed_flags",
            "agent_slo_compliant",
            "agent_fr",
            "agent_minimum_distance_recovery",
            "agent_excess_changed_flags",
            "match",
            "regret",
        }
        missing = sorted(required.difference(ddqn_details.columns))
        if missing:
            raise ValueError(
                f"DDQN details CSV is missing comparison columns: {missing}"
            )

        if len(ddqn_details) != len(ga_details):
            raise ValueError(
                "DDQN and GA evaluation details have different row counts: "
                f"{len(ddqn_details)} vs {len(ga_details)}"
            )

        available_alignment = [
            key
            for key in ALIGNMENT_KEYS
            if key in ddqn_details.columns and key in ga_details.columns
        ]
        if available_alignment:
            left = ga_details[available_alignment].astype(str).reset_index(drop=True)
            right = ddqn_details[available_alignment].astype(str).reset_index(drop=True)
            if not left.equals(right):
                raise ValueError(
                    "DDQN details are not aligned with the GA evaluation states."
                )

        rows.insert(0, summarize_ddqn(ddqn_details, ddqn_alpha))

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "ddqn_ga_oracle_comparison.csv", index=False)

    if ddqn_details is not None:
        save_regret_histogram(
            ddqn_regret=ddqn_details["regret"].astype(float),
            ga_regret=ga_details["ga_regret"].astype(float),
            output_dir=output_dir,
        )

    return comparison


def save_regret_histogram(
    ddqn_regret: pd.Series,
    ga_regret: pd.Series,
    output_dir: Path,
) -> None:
    maximum = max(float(ddqn_regret.max()), float(ga_regret.max()), 0.01)
    bins = np.linspace(0.0, maximum, 30)
    width = bins[1] - bins[0]
    centers = (bins[:-1] + bins[1:]) / 2.0
    ddqn_counts, _ = np.histogram(ddqn_regret, bins=bins)
    ga_counts, _ = np.histogram(ga_regret, bins=bins)

    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.bar(
        centers - width / 4.0,
        ddqn_counts,
        width=width / 2.0,
        label="DDQN",
        edgecolor="black",
        linewidth=0.6,
    )
    axis.bar(
        centers + width / 4.0,
        ga_counts,
        width=width / 2.0,
        label="DAGAME-GA",
        edgecolor="black",
        linewidth=0.6,
    )
    axis.set_xlabel("Regret relative to exhaustive oracle")
    axis.set_ylabel("Number of violation states")
    axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ddqn_ga_regret_histogram.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "ddqn_ga_regret_histogram.pdf", bbox_inches="tight")
    plt.close(fig)


def configure_logging(output_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "ga_training.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir)
    set_global_seed(args.seed)

    metrics_df = pd.read_csv(args.metrics)
    configs_df = pd.read_csv(args.configs)
    evaluation_df = pd.read_csv(args.evaluation)

    tau_ms, validation_report = validate_inputs(
        metrics_df=metrics_df,
        configs_df=configs_df,
        evaluation_df=evaluation_df,
        explicit_tau=args.tau,
    )
    validation_report.update(
        {
            "alpha": float(args.alpha),
            "beta": float(args.beta),
            "ga_seed": int(args.seed),
            "population_size": int(args.pop),
            "generations": int(args.gens),
            "crossover_probability": float(args.crossover_probability),
            "mutation_probability": float(args.mutation_probability),
        }
    )
    with (output_dir / "input_validation.json").open("w", encoding="utf-8") as file:
        json.dump(json_safe(validation_report), file, indent=2)

    print("\nInput validation:")
    print(json.dumps(json_safe(validation_report), indent=2))

    if args.validate_only:
        print("\nValidation completed; models were not loaded and GA was not executed.")
        return

    rt_preprocessor = joblib.load(args.rt_preprocessor)
    rt_model = joblib.load(args.rt_model)

    problem = TransitionAwareProblem(
        metrics_df=metrics_df,
        configs_df=configs_df,
        tau_ms=tau_ms,
        rt_preprocessor=rt_preprocessor,
        rt_model=rt_model,
        alpha=args.alpha,
        beta=args.beta,
    )
    ga = DAGAMEGA(
        problem=problem,
        population_size=args.pop,
        generations=args.gens,
        crossover_probability=args.crossover_probability,
        mutation_probability=args.mutation_probability,
    )

    started = time.time()
    details, summary = evaluate_ga(
        problem=problem,
        ga=ga,
        evaluation_df=evaluation_df,
        base_seed=args.seed,
        output_dir=output_dir,
    )
    elapsed = float(time.time() - started)

    ddqn_path = Path(args.ddqn_details) if args.ddqn_details else None
    comparison = create_comparison(
        ga_details=details,
        ddqn_details_path=ddqn_path,
        ddqn_alpha=args.ddqn_alpha,
        output_dir=output_dir,
    )

    run_summary = {
        **validation_report,
        "total_ga_evaluation_time_seconds": elapsed,
        "ga_summary": summary.iloc[0].to_dict(),
        "comparison_file_created": bool(comparison is not None),
        "ddqn_details_used": str(ddqn_path) if ddqn_path is not None else None,
    }
    with (output_dir / "ga_run_summary.json").open("w", encoding="utf-8") as file:
        json.dump(json_safe(run_summary), file, indent=2)

    print("\nDAGAME-GA evaluation summary:")
    print(summary.to_string(index=False))
    if comparison is not None:
        print("\nDDQN / GA / Oracle comparison:")
        print(comparison.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transition-aware DAGAME-style GA baseline for violation-only JNN "
            "tail-latency recovery."
        )
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
    parser.add_argument(
        "--ddqn-details",
        default=None,
        help=(
            "Optional alpha=0.90 DDQN evaluation_details.csv. When supplied, "
            "the script creates a DDQN/GA/oracle comparison table and regret plot."
        ),
    )
    parser.add_argument("--ddqn-alpha", type=float, default=0.90)
    parser.add_argument("--output-dir", default="Results_violation/ga_alpha_0.90_seed_45")
    parser.add_argument("--alpha", type=float, default=0.90)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--gens", type=int, default=10)
    parser.add_argument("--crossover-probability", type=float, default=1.0)
    parser.add_argument("--mutation-probability", type=float, default=1.0)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs without loading PPM models or running GA.",
    )
    return parser


if __name__ == "__main__":
    run(build_argument_parser().parse_args())
