import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def parse_args():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("data_frame_path", type=str, help="Path to dataframe")
    parser.add_argument(
        "--prediction",
        type=str,
        default="experimenter",
        help="Which attribute to predict",
    )
    parser.add_argument(
        "--features", nargs="+", default=["chirp_8Hz_average_norm", "preproc_bar"], help="Input features"
    )
    parser.add_argument("--fft", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experimenters",
        nargs="+",
        default=None,
        help="Filter by the given experimenters. Specify exactly two experimenters if you want to classify them against each other",
    )
    parser.add_argument(
        "--location",
        choices=["temporal_ventral", "temporal_dorsal", "nasal_ventral", "nasal_dorsal", "best"],
        default=None,
    )
    parser.add_argument("--celltype", choices=list(range(1, 58)), type=int, default=None)
    parser.add_argument("--lowpass-filter", type=float, default=None)
    parser.add_argument("--field", type=int, default=None)
    parser.add_argument("--do-not-quality-filter", action=argparse.BooleanOptionalAction)
    parser.add_argument("--supergroup", type=str, default=None)

    return parser.parse_args()


def predictions_summed_over_unique_ids(
    probs: np.ndarray,
    targets: np.ndarray,
    identifiers: list[str],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Averages predictions over the specified identifiers.
    If the identifiers do not have a unique target, throw an error.
    (e.g. when we predict ventral vs dorsal, but there are cells that are both ventral and dorsal in a field)
    """
    preds = []
    unique_ids = sorted(set(identifiers))
    unique_targets = []
    for id_ in unique_ids:
        traces_in_id = np.array([id_ == loop_id for loop_id in identifiers])
        indices = [i for i, b in enumerate(traces_in_id) if b]
        tgt = np.unique(targets[indices])
        if len(tgt) != 1:
            raise ValueError(f"Cannot average, multiple target for identifier {id_}: {tgt}")
        unique_targets.append(tgt[0])
        summed_probs = (probs * np.expand_dims(traces_in_id, 1)).sum(0)
        prediction = np.argmax(summed_probs)
        preds.append(prediction)

    preds_np = np.array(preds)
    tgt_np = np.array(unique_targets)
    accuracy = (preds_np == tgt_np).sum() / preds_np.shape[0]

    return accuracy, preds_np, tgt_np


def predictions_averaged_over_unique_ids(
    predictions: np.ndarray,
    targets: np.ndarray,
    identifiers: list[str],
) -> tuple[float, float, np.ndarray, np.ndarray]:
    avg_preds = []
    unique_ids = sorted(set(identifiers))
    avg_targets = []
    for id_ in unique_ids:
        traces_in_id = np.array([id_ == loop_id for loop_id in identifiers])
        avg_prediction = np.sum((predictions * traces_in_id) / np.sum(traces_in_id))
        avg_target = np.sum((targets * traces_in_id) / np.sum(traces_in_id))
        avg_preds.append(avg_prediction)
        avg_targets.append(avg_target)

    preds_np = np.array(avg_preds)
    tgt_np = np.array(avg_targets)
    error_per_identifier = np.abs(tgt_np - preds_np)
    error = np.average(error_per_identifier)
    mean_squared_error = np.average(error_per_identifier**2)

    return error, mean_squared_error, preds_np, tgt_np


def get_results_file_path(
    dataset_name: str,
    method_name: str,
    use_fft: bool,
    experimenters: list[str] | None,
    location: str | None,
    celltype: int | None,
    lowpass_filter: float | None,
    features: list[str] | None,
    do_not_quality_filter: bool,
    field: int | None,
    supergroup: str | None,
) -> str:
    results_path = f"results/{dataset_name}_{method_name}"
    if use_fft:
        results_path += "_fft"
    if experimenters is not None:
        results_path += f"_{'-'.join(experimenters)}"
    if location is not None:
        results_path += f"_{location}"
    if celltype is not None:
        results_path += f"_groupid{celltype:02d}"
    if supergroup is not None:
        supergroup_str = supergroup.replace(" ", "-")
        results_path += f"_supergroup_{supergroup_str}"
    if lowpass_filter is not None:
        results_path += f"_lowpass_{lowpass_filter:.2f}"
    if field is not None:
        results_path += f"field_id_{field}"
    if features is not None:
        results_path += f"_features_{'_'.join(features)}"
    if do_not_quality_filter:
        results_path += "do_not_quality_filter"

    results_path += ".csv"
    return results_path


def save_confusion_matrix(test_data, y_pred_ids: np.ndarray, dataset_name: str, postfix: str):
    y_pred = [test_data.ID_TO_LABEL[x] for x in y_pred_ids]
    y_true = [test_data.ID_TO_LABEL[x] for x in test_data.targets_np()]
    cm_display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    cm_display.plot()
    folder = f"results/{dataset_name}/"
    os.makedirs(folder, exist_ok=True)
    plt.gcf().savefig(folder + f"confusion_matrix_{postfix}.jpg")


def save_prediction_probs(probs: np.ndarray, dataset_name: str, postfix: str) -> None:
    np.save(f"results/{dataset_name}/predicted_probs_{postfix}.npy", probs)
