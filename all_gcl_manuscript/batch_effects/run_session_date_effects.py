#!/usr/bin/env python3

import argparse
import csv
import itertools
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas
import torch
from sklearn.ensemble import RandomForestClassifier

from all_gcl_manuscript.batch_effects.data import filter_data, load_data
from all_gcl_manuscript.batch_effects.datasets import BaseDataset


DEFAULT_FEATURES = ["chirp_8Hz_average_norm", "preproc_bar"]
DEFAULT_FIELD_IDS = (0, 1)
DEFAULT_EXP_NUM = 1
DEFAULT_MAX_AGE_DIFF_WEEKS = 10.0
DEFAULT_MAX_FIELD_DISTANCE_UM = -1.0
DEFAULT_MATCH_RETINAL_QUADRANT = False
RESULT_FIELDNAMES = [
    "seed",
    "exp_num",
    "retinal_quadrant",
    "experimenter",
    "animgender",
    "animgender_a",
    "animgender_b",
    "age_week",
    "age_week_a",
    "age_week_b",
    "age_week_diff",
    "field_0_distance_um",
    "field_1_distance_um",
    "field_2_distance_um",
    "field_distance_um_mean",
    "field_distance_um_max",
    "date_a",
    "date_b",
    "train_field_id",
    "test_field_id",
    "num_train",
    "num_test",
    "num_train_fields",
    "num_test_fields",
    "num_train_retinas",
    "num_test_retinas",
    "train_accuracy_cell",
    "test_accuracy_cell",
    "test_accuracy_field",
    "test_accuracy_retina",
]

# Keep the model identical to the existing random-forest batch-effect analyses.
CLASSIFIER_CONFIG = {
    "class_weight": "balanced",
    "random_state": 42,
    "oob_score": True,
    "ccp_alpha": 0.00021870687842726034,
    "max_depth": 50,
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0,
    "n_estimators": 1000,
    "n_jobs": 20,
}


class SessionDateBinaryDataset(BaseDataset):
    LABELS = ["date_a", "date_b"]
    ID_TO_LABEL = dict(enumerate(LABELS))
    LABEL_TO_ID = {label: idx for idx, label in ID_TO_LABEL.items()}

    def __init__(
        self,
        dataframe: pandas.DataFrame,
        input_names: Iterable[str],
        use_fft_features: bool = False,
        low_pass_filter_frequency: float | None = None,
    ):
        targets = torch.LongTensor(dataframe["date_binary"].values)
        self.num_categories = len(self.LABELS)
        super().__init__(dataframe, input_names, targets, use_fft_features, low_pass_filter_frequency)

    def should_use_classification(self) -> bool:
        return True


@dataclass(frozen=True)
class DatePair:
    match_values: tuple[tuple[str, Any], ...]
    date_a: str
    date_b: str
    animgender_a: str
    animgender_b: str
    age_week_a: float
    age_week_b: float
    field_distances_um: tuple[tuple[int, float], ...]

    def get_match_value(self, column: str, default: Any = None) -> Any:
        return dict(self.match_values).get(column, default)

    def metadata(self) -> dict[str, Any]:
        age_week = self.get_match_value("age_week", "")
        age_week_diff = abs(self.age_week_a - self.age_week_b)
        field_distances = dict(self.field_distances_um)
        finite_distances = [distance for distance in field_distances.values() if np.isfinite(distance)]
        return {
            "retinal_quadrant": self.get_match_value("retinal_quadrant", "any"),
            "experimenter": self.get_match_value("experimenter"),
            "animgender": self.get_match_value("animgender", "any"),
            "animgender_a": self.animgender_a,
            "animgender_b": self.animgender_b,
            "age_week": age_week,
            "age_week_a": self.age_week_a,
            "age_week_b": self.age_week_b,
            "age_week_diff": age_week_diff,
            "field_0_distance_um": field_distances.get(0, ""),
            "field_1_distance_um": field_distances.get(1, ""),
            "field_2_distance_um": field_distances.get(2, ""),
            "field_distance_um_mean": float(np.mean(finite_distances)) if finite_distances else "",
            "field_distance_um_max": max(finite_distances) if finite_distances else "",
            "date_a": self.date_a,
            "date_b": self.date_b,
        }


@dataclass(frozen=True)
class TrainableDirection:
    date_pair: DatePair
    train_field_id: int
    test_field_id: int
    train_frame: pandas.DataFrame
    test_frame: pandas.DataFrame


@dataclass(frozen=True)
class SkippedDirection:
    date_pair: DatePair
    train_field_id: int
    test_field_id: int
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate recording-session batch effects by asking whether a random forest can distinguish "
            "matched recording dates while generalizing across corrected scan fields."
        )
    )
    parser.add_argument("data_frame_path", type=str, help="Path to dataframe")
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="Input features. Defaults match the existing batch-effect analyses.",
    )
    parser.add_argument("--fft", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--field-ids",
        nargs=2,
        type=int,
        default=DEFAULT_FIELD_IDS,
        help="Corrected field IDs. The primary experiment uses field IDs 0 and 1 in code.",
    )
    parser.add_argument(
        "--min-cells-per-date-field",
        type=int,
        default=30,
        help="Minimum number of cells required for each date in each of the two corrected fields.",
    )
    parser.add_argument(
        "--experimenters",
        nargs="+",
        default=None,
        help="Optional experimenter filter.",
    )
    parser.add_argument(
        "--location",
        choices=["temporal_ventral", "temporal_dorsal", "nasal_ventral", "nasal_dorsal", "best"],
        default=None,
    )
    parser.add_argument("--celltype", choices=list(range(1, 58)), type=int, default=None)
    parser.add_argument("--lowpass-filter", type=float, default=None)
    parser.add_argument("--do-not-quality-filter", action=argparse.BooleanOptionalAction)
    parser.add_argument("--supergroup", type=str, default=None)
    parser.add_argument(
        "--exp-num",
        choices=[1, 2],
        type=int,
        default=DEFAULT_EXP_NUM,
        help="Restrict to one experiment number. Defaults to exp_num 1 for the first eye/retina.",
    )
    parser.add_argument(
        "--match-sex",
        action="store_true",
        help="Require the two recording dates to have the same animal sex.",
    )
    parser.add_argument(
        "--match-age-week",
        action="store_true",
        help="Require the two recording dates to have the same rounded age week.",
    )
    parser.add_argument(
        "--max-age-diff-weeks",
        type=float,
        default=DEFAULT_MAX_AGE_DIFF_WEEKS,
        help=(
            "Maximum allowed age difference between paired recording dates. "
            "Ignored when --match-age-week is set. Use a negative value to disable this filter."
        ),
    )
    parser.add_argument(
        "--max-field-distance-um",
        type=float,
        default=DEFAULT_MAX_FIELD_DISTANCE_UM,
        help=(
            "Maximum allowed Euclidean distance between same-ID field centers across paired dates. "
            "Use a negative value to disable this filter."
        ),
    )
    parser.add_argument(
        "--match-retinal-quadrant",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MATCH_RETINAL_QUADRANT,
        help="Require the two recording dates to be in the same retinal quadrant.",
    )
    parser.add_argument(
        "--balance-columns",
        nargs="*",
        default=None,
        help=(
            "Optional metadata columns for stratified downsampling within each date class, "
            "for example: --balance-columns supergroup"
        ),
    )
    parser.add_argument(
        "--permute-labels",
        action="store_true",
        help=(
            "Run a null baseline by shuffling training date labels after balancing. "
            "Held-out test labels are kept unchanged so field/retina-level scoring remains valid."
        ),
    )
    parser.add_argument(
        "--max-date-pairs",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Where to append result rows. Defaults to results/session_date_binary_*.csv.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print eligible matched date pairs without training models.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-direction training and skip details.",
    )
    return parser.parse_args()


def normalize_date(value) -> str:
    if not isinstance(value, str) and hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)


def raw_field_id(field: str) -> int | None:
    match = re.search(r"\d+", str(field))
    return int(match.group(0)) if match is not None else None


def add_retinal_quadrant(data: pandas.DataFrame) -> pandas.DataFrame:
    data = data.copy()
    conditions = [
        (data["ventral_dorsal_pos_um"] < 0) & (data["temporal_nasal_pos_um"] < 0),
        (data["ventral_dorsal_pos_um"] > 0) & (data["temporal_nasal_pos_um"] < 0),
        (data["ventral_dorsal_pos_um"] < 0) & (data["temporal_nasal_pos_um"] > 0),
        (data["ventral_dorsal_pos_um"] > 0) & (data["temporal_nasal_pos_um"] > 0),
    ]
    labels = ["temporal_ventral", "temporal_dorsal", "nasal_ventral", "nasal_dorsal"]
    data["retinal_quadrant"] = np.select(conditions, labels, default="unknown")
    return data


def add_corrected_field_id(data: pandas.DataFrame) -> pandas.DataFrame:
    """Add a field_id column using the same relative numbering idea as create_field_id_binary."""
    field_maps: dict[tuple[str, str, int], dict[str, int]] = {}
    data_with_fields = data[data["field"].notna()]

    for (experimenter, date, exp_num), data_exp in data_with_fields.groupby(["experimenter", "date", "exp_num"]):
        fields = sorted(set(data_exp["field"]))
        field_ids = [raw_field_id(field) for field in fields]
        if None in field_ids:
            continue
        if any(str(field).endswith(("a", "b")) for field in fields):
            continue
        min_field_id = min(field_ids)
        max_field_id = max(field_ids)
        if min_field_id > 1:
            continue
        if set(range(min_field_id, max_field_id + 1)) != set(field_ids):
            continue

        field_maps[(experimenter, date, int(exp_num))] = {
            field: field_id - min_field_id for field, field_id in zip(fields, field_ids, strict=True)
        }

    def corrected_field_id(row) -> int:
        key = (row["experimenter"], row["date"], int(row["exp_num"]))
        return field_maps.get(key, {}).get(row["field"], -1)

    data = data.copy()
    data["field_id"] = data.apply(corrected_field_id, axis=1)
    return data


def prepare_data(args: argparse.Namespace) -> pandas.DataFrame:
    data = load_data(args.data_frame_path)
    chirp_threshold, bar_threshold = (0.0, 0.0) if args.do_not_quality_filter else (0.35, 0.6)
    data = filter_data(
        data,
        chirp_threshold,
        bar_threshold,
        location=args.location,
        celltype=args.celltype,
        setup_id="1",
        supergroup=args.supergroup,
    )
    data = data.copy()
    data["date"] = data["date"].map(normalize_date)
    if args.experimenters is not None:
        data = data[data["experimenter"].isin(args.experimenters)]
    data = data[data["exp_num"] == args.exp_num]
    data = add_retinal_quadrant(data)
    if args.match_retinal_quadrant:
        data = data[data["retinal_quadrant"] != "unknown"]

    data = add_corrected_field_id(data)
    data = data[data["field_id"].isin(args.field_ids)].copy()

    data["age_days"] = pandas.to_numeric(data["age"], errors="coerce")
    needs_age = args.match_age_week or args.max_age_diff_weeks >= 0
    if needs_age:
        data = data[(data["age_days"] > 5) & (data["age_days"] < 200)]
    if args.match_sex:
        data = data[data["animgender"].isin(["female", "male"])]
    if data["age_days"].notna().all():
        data["age_week"] = (data["age_days"] / 7).round().astype(int)
    else:
        data["age_week"] = (data["age_days"] / 7).round()
    return data


def get_match_columns(args: argparse.Namespace) -> list[str]:
    match_columns = ["experimenter"]
    if args.match_sex:
        match_columns.append("animgender")
    if args.match_age_week:
        match_columns.append("age_week")
    if args.match_retinal_quadrant:
        match_columns.append("retinal_quadrant")
    return match_columns


def summarize_date_group(data_date: pandas.DataFrame) -> dict[str, Any]:
    animgenders = sorted(str(value) for value in data_date["animgender"].dropna().unique())
    field_centers_um = {}
    for field_id, data_field in data_date.groupby("field_id"):
        temporal_nasal_values = pandas.to_numeric(data_field["field_temporal_nasal_pos_um"], errors="coerce")
        ventral_dorsal_values = pandas.to_numeric(data_field["field_ventral_dorsal_pos_um"], errors="coerce")
        if temporal_nasal_values.notna().any() and ventral_dorsal_values.notna().any():
            field_centers_um[int(field_id)] = (
                float(temporal_nasal_values.median()),
                float(ventral_dorsal_values.median()),
            )
    return {
        "date": data_date["date"].iloc[0],
        "age_week": float(data_date["age_week"].median()),
        "animgender": "/".join(animgenders) if animgenders else "unknown",
        "field_centers_um": field_centers_um,
    }


def compute_field_distances_um(
    date_a: dict[str, Any],
    date_b: dict[str, Any],
    field_ids: Sequence[int],
) -> tuple[tuple[int, float], ...] | None:
    distances = []
    centers_a = date_a["field_centers_um"]
    centers_b = date_b["field_centers_um"]
    for field_id in field_ids:
        if field_id not in centers_a or field_id not in centers_b:
            return None
        center_a = np.array(centers_a[field_id], dtype=float)
        center_b = np.array(centers_b[field_id], dtype=float)
        distance = float(np.linalg.norm(center_a - center_b))
        if not np.isfinite(distance):
            return None
        distances.append((int(field_id), distance))
    return tuple(distances)


def iter_date_pairs(
    data: pandas.DataFrame,
    args: argparse.Namespace,
) -> list[DatePair]:
    date_pairs = []
    match_columns = get_match_columns(args)
    max_age_diff_weeks = None if args.match_age_week or args.max_age_diff_weeks < 0 else args.max_age_diff_weeks
    max_field_distance_um = None if args.max_field_distance_um < 0 else args.max_field_distance_um

    for match_values, data_matched in data.groupby(match_columns, dropna=False):
        if len(match_columns) == 1 and not isinstance(match_values, tuple):
            match_values = (match_values,)
        match_dict = dict(zip(match_columns, match_values, strict=True))
        usable_dates = []
        for _, data_date in data_matched.groupby("date"):
            counts = data_date.groupby("field_id").size()
            has_enough_cells = all(
                counts.get(field_id, 0) >= args.min_cells_per_date_field for field_id in args.field_ids
            )
            if has_enough_cells:
                usable_dates.append(summarize_date_group(data_date))

        for date_a, date_b in itertools.combinations(usable_dates, 2):
            age_week_diff = abs(date_a["age_week"] - date_b["age_week"])
            if max_age_diff_weeks is not None and age_week_diff > max_age_diff_weeks:
                continue
            field_distances_um = compute_field_distances_um(date_a, date_b, args.field_ids)
            if max_field_distance_um is not None:
                if field_distances_um is None:
                    continue
                if max(distance for _, distance in field_distances_um) > max_field_distance_um:
                    continue
            date_pairs.append(
                DatePair(
                    match_values=tuple(match_dict.items()),
                    date_a=str(date_a["date"]),
                    date_b=str(date_b["date"]),
                    animgender_a=date_a["animgender"],
                    animgender_b=date_b["animgender"],
                    age_week_a=date_a["age_week"],
                    age_week_b=date_b["age_week"],
                    field_distances_um=field_distances_um or tuple(),
                )
            )
    return date_pairs


def balanced_sample(
    data: pandas.DataFrame,
    target_column: str,
    seed: int,
    balance_columns: Sequence[str] | None = None,
) -> pandas.DataFrame:
    target_values = sorted(data[target_column].unique())
    if len(target_values) != 2:
        return data.iloc[0:0].copy()

    if not balance_columns:
        min_count = data[target_column].value_counts().min()
        return (
            data.groupby(target_column, group_keys=False)
            .sample(n=min_count, random_state=seed)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )

    samples = []
    for _, data_stratum in data.groupby(list(balance_columns), dropna=False):
        counts = data_stratum[target_column].value_counts()
        if all(counts.get(target_value, 0) > 0 for target_value in target_values):
            min_count = counts.min()
            samples.append(
                data_stratum.groupby(target_column, group_keys=False).sample(n=min_count, random_state=seed)
            )

    if not samples:
        return data.iloc[0:0].copy()
    return pandas.concat(samples).sample(frac=1, random_state=seed).reset_index(drop=True)


def get_date_pair_frame(data: pandas.DataFrame, date_pair: DatePair) -> pandas.DataFrame:
    mask = data["date"].isin([date_pair.date_a, date_pair.date_b])
    for column, value in date_pair.match_values:
        mask &= data[column] == value
    data_pair = data[mask].copy()
    data_pair["date_binary"] = (data_pair["date"] == date_pair.date_b).astype(int)
    return data_pair


def validate_split(train_frame: pandas.DataFrame, test_frame: pandas.DataFrame) -> str | None:
    if len(train_frame) == 0:
        return "train split is empty after balancing"
    if len(test_frame) == 0:
        return "test split is empty after balancing"
    if train_frame["date_binary"].nunique() != 2:
        return "train split has fewer than two date classes after balancing"
    if test_frame["date_binary"].nunique() != 2:
        return "test split has fewer than two date classes after balancing"
    return None


def permute_train_labels(train_frame: pandas.DataFrame, seed: int) -> pandas.DataFrame:
    train_frame = train_frame.copy()
    rng = np.random.default_rng(seed)
    train_frame["date_binary"] = rng.permutation(train_frame["date_binary"].to_numpy())
    return train_frame


def build_train_test_frames(
    data_pair: pandas.DataFrame,
    train_field_id: int,
    test_field_id: int,
    args: argparse.Namespace,
) -> tuple[pandas.DataFrame, pandas.DataFrame, str | None]:
    sample_seed = args.seed + 17 * train_field_id + 31 * test_field_id
    train_frame = balanced_sample(
        data_pair[data_pair["field_id"] == train_field_id],
        target_column="date_binary",
        seed=sample_seed,
        balance_columns=args.balance_columns,
    )
    test_frame = balanced_sample(
        data_pair[data_pair["field_id"] == test_field_id],
        target_column="date_binary",
        seed=sample_seed + 1,
        balance_columns=args.balance_columns,
    )
    reason = validate_split(train_frame, test_frame)
    if reason is None and args.permute_labels:
        train_frame = permute_train_labels(train_frame, seed=sample_seed + 1009)
    return train_frame, test_frame, reason


def get_trainable_directions(
    data: pandas.DataFrame,
    date_pairs: Sequence[DatePair],
    args: argparse.Namespace,
) -> tuple[list[TrainableDirection], list[SkippedDirection]]:
    trainable_directions = []
    skipped_directions = []
    for date_pair in date_pairs:
        data_pair = get_date_pair_frame(data, date_pair)
        for train_field_id, test_field_id in (tuple(args.field_ids), tuple(reversed(args.field_ids))):
            if len(data_pair) == 0:
                skipped_directions.append(
                    SkippedDirection(
                        date_pair=date_pair,
                        train_field_id=train_field_id,
                        test_field_id=test_field_id,
                        reason="no rows match this date-pair metadata after filtering",
                    )
                )
                continue

            train_frame, test_frame, reason = build_train_test_frames(
                data_pair, train_field_id, test_field_id, args
            )
            if reason is not None:
                skipped_directions.append(
                    SkippedDirection(
                        date_pair=date_pair,
                        train_field_id=train_field_id,
                        test_field_id=test_field_id,
                        reason=reason,
                    )
                )
                continue
            trainable_directions.append(
                TrainableDirection(
                    date_pair=date_pair,
                    train_field_id=train_field_id,
                    test_field_id=test_field_id,
                    train_frame=train_frame,
                    test_frame=test_frame,
                )
            )
    return trainable_directions, skipped_directions


def predictions_summed_over_unique_ids(
    probabilities: np.ndarray,
    targets: np.ndarray,
    identifiers: list[str],
) -> tuple[float, np.ndarray, np.ndarray]:
    predictions = []
    unique_targets = []
    for identifier in sorted(set(identifiers)):
        traces_in_identifier = np.array([identifier == loop_identifier for loop_identifier in identifiers])
        indices = [idx for idx, is_in_identifier in enumerate(traces_in_identifier) if is_in_identifier]
        target = np.unique(targets[indices])
        if len(target) != 1:
            raise ValueError(f"Cannot average, multiple targets for identifier {identifier}: {target}")
        unique_targets.append(target[0])
        summed_probabilities = (probabilities * np.expand_dims(traces_in_identifier, 1)).sum(0)
        predictions.append(np.argmax(summed_probabilities))

    predictions_np = np.array(predictions)
    targets_np = np.array(unique_targets)
    accuracy = (predictions_np == targets_np).sum() / predictions_np.shape[0]
    return accuracy, predictions_np, targets_np


def score_model(model: RandomForestClassifier, data: SessionDateBinaryDataset) -> tuple[float, float, float]:
    probabilities = model.predict_proba(data.inputs_np())
    accuracy_cell = model.score(X=data.inputs_np(), y=data.targets_np())
    class_to_model_class = {
        original_class: model_class for model_class, original_class in enumerate(model.classes_)
    }
    remapped_targets = np.array([class_to_model_class[target] for target in data.targets_np()])
    accuracy_field, _, _ = predictions_summed_over_unique_ids(
        probabilities, remapped_targets, data.get_field_ids()
    )
    accuracy_retina, _, _ = predictions_summed_over_unique_ids(
        probabilities, remapped_targets, data.get_retina_ids()
    )
    return accuracy_cell, accuracy_field, accuracy_retina


def train_one_direction(
    direction: TrainableDirection,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_data = SessionDateBinaryDataset(
        direction.train_frame,
        input_names=args.features,
        use_fft_features=args.fft,
        low_pass_filter_frequency=args.lowpass_filter,
    )
    test_data = SessionDateBinaryDataset(
        direction.test_frame,
        input_names=args.features,
        use_fft_features=args.fft,
        low_pass_filter_frequency=args.lowpass_filter,
    )

    model = RandomForestClassifier(**CLASSIFIER_CONFIG)
    model.fit(X=train_data.inputs_np(), y=train_data.targets_np())

    train_accuracy_cell = model.score(X=train_data.inputs_np(), y=train_data.targets_np())
    test_accuracy_cell, test_accuracy_field, test_accuracy_retina = score_model(model, test_data)

    return {
        "seed": args.seed,
        "exp_num": args.exp_num,
        **direction.date_pair.metadata(),
        "train_field_id": direction.train_field_id,
        "test_field_id": direction.test_field_id,
        "num_train": len(train_data),
        "num_test": len(test_data),
        "num_train_fields": len(set(train_data.get_field_ids())),
        "num_test_fields": len(set(test_data.get_field_ids())),
        "num_train_retinas": len(set(train_data.get_retina_ids())),
        "num_test_retinas": len(set(test_data.get_retina_ids())),
        "train_accuracy_cell": train_accuracy_cell,
        "test_accuracy_cell": test_accuracy_cell,
        "test_accuracy_field": test_accuracy_field,
        "test_accuracy_retina": test_accuracy_retina,
    }


def get_output_path(args: argparse.Namespace) -> str:
    if args.output_path is not None:
        return args.output_path

    features = "_".join(args.features)
    field_ids = "-".join(map(str, args.field_ids))
    label_mode = "_train_labels_permuted" if args.permute_labels else ""
    match_parts = get_match_columns(args)
    if not args.match_age_week:
        if args.max_age_diff_weeks < 0:
            match_parts.append("age_unmatched")
        else:
            age_diff_str = f"{args.max_age_diff_weeks:g}".replace(".", "p")
            match_parts.append(f"age_diff_le_{age_diff_str}w")
    if not args.match_sex:
        match_parts.append("sex_unmatched")
    if args.max_field_distance_um < 0:
        match_parts.append("field_dist_unmatched")
    else:
        field_dist_str = f"{args.max_field_distance_um:g}".replace(".", "p")
        match_parts.append(f"field_dist_le_{field_dist_str}um")
    match_str = "-".join(match_parts)
    return (
        f"results/session_date_binary_exp_num_{args.exp_num}_match_{match_str}_field_ids_{field_ids}"
        f"_features_{features}{label_mode}.csv"
    )


def append_result(row: dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=RESULT_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    data = prepare_data(args)
    date_pairs = iter_date_pairs(data, args)
    if args.max_date_pairs is not None:
        date_pairs = date_pairs[: args.max_date_pairs]
    trainable_directions, skipped_directions = get_trainable_directions(data, date_pairs, args)

    print(
        f"Found {len(date_pairs)} eligible date pairs "
        f"({2 * len(date_pairs)} field directions) for exp_num {args.exp_num} "
        f"and corrected field IDs {tuple(args.field_ids)} using match columns {get_match_columns(args)}."
    )
    if not args.match_age_week and args.max_age_diff_weeks >= 0:
        print(f"Maximum paired-date age difference: {args.max_age_diff_weeks:g} weeks.")
    if args.max_field_distance_um >= 0:
        print(f"Maximum same-field center distance between paired dates: {args.max_field_distance_um:g} um.")
    if args.permute_labels:
        print("Permutation/null mode: training date labels are shuffled after balancing.")
    print(
        f"{len(trainable_directions)} directions are trainable; "
        f"{len(skipped_directions)} skipped after final balancing."
    )
    if skipped_directions and not args.verbose:
        print("Use --verbose to print skip reasons.")
    if args.verbose:
        for skipped_direction in skipped_directions:
            print(
                "Skipping "
                f"{skipped_direction.date_pair.date_a} vs {skipped_direction.date_pair.date_b}, "
                f"train field {skipped_direction.train_field_id}, "
                f"test field {skipped_direction.test_field_id}: "
                f"{skipped_direction.reason}."
            )

    if args.list_only:
        for date_pair in date_pairs:
            print(date_pair)
        return

    output_path = get_output_path(args)
    if not trainable_directions:
        print(f"No classifiers trained; no results appended to {os.path.abspath(output_path)}")
        return

    print(f"Training {len(trainable_directions)} session-date classifiers...")
    for direction_idx, direction in enumerate(trainable_directions, start=1):
        if args.verbose:
            print(
                "Training session-date classifier "
                f"{direction.date_pair.date_a} vs {direction.date_pair.date_b}, "
                f"train field {direction.train_field_id}, test field {direction.test_field_id}"
            )
        row = train_one_direction(direction, args)
        append_result(row, output_path)
        if args.verbose:
            print(
                f"test_accuracy_cell={row['test_accuracy_cell']:.1%} "
                f"test_accuracy_field={row['test_accuracy_field']:.1%} "
                f"test_accuracy_retina={row['test_accuracy_retina']:.1%}"
            )
        elif direction_idx % 10 == 0 or direction_idx == len(trainable_directions):
            print(f"Trained {direction_idx}/{len(trainable_directions)} classifiers.")

    print(f"Appended results to {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
