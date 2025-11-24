#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from all_gcl_manuscript.batch_effects.data import get_dataset_split_from_args
from all_gcl_manuscript.batch_effects.random_forest import (
    get_results_file_path,
    parse_args,
    predictions_averaged_over_unique_ids,
    predictions_summed_over_unique_ids,
    save_confusion_matrix,
    save_prediction_probs,
)

# From djimaging:
# https://github.com/eulerlab/djimaging/blob/e5a5b1f0366d68992718076a0aae541fe7309c0e/djimaging/tables/classifier/rgc_classifier.py#L163
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
REGRESSOR_CONFIG = {
    "random_state": 42,
    "oob_score": True,
    "ccp_alpha": 0.00021870687842726034,
    "max_depth": 50,
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0,
    "n_estimators": 1000,
    "n_jobs": 20,
}


def plot_importances(random_forest) -> plt.Figure:
    importances = random_forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in random_forest.estimators_], axis=0)
    feature_names = [f"feature {i}" for i in range(importances.shape[0])]

    forest_importances = pandas.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    return fig


def train(
    train_data,
    test_data,
    results_path: str,
    seed: int,
    dataset_name: str,
    plot_traces: bool = False,
    do_save_confusion_matrix: bool = False,
    export_random_forest_vis: bool = False,
    do_plot_feature_importances: bool = False,
    output_accuracies_grouped_by_celltype: bool = False,
) -> None:
    if len(results_path) > 200:
        results_path = results_path[:190] + "_etal.csv"
    if train_data.should_use_classification():
        model = RandomForestClassifier(**CLASSIFIER_CONFIG)
    else:
        model = RandomForestRegressor(**REGRESSOR_CONFIG)

    print("Training random forest...")
    model.fit(X=train_data.inputs_np(), y=train_data.targets_np())

    for name, data in [("train", train_data), ("test", test_data)]:
        print(f"{name.upper()}: {repr(data)}")
        print(f"{name.upper()}: {str(data)}")
        score = model.score(X=data.inputs_np(), y=data.targets_np())
        print(f"Score {name}: {score:.1%}")

    # Sum by fields
    field_ids = test_data.get_field_ids()
    retina_ids = test_data.get_retina_ids()
    if train_data.should_use_classification():
        test_probs = model.predict_proba(test_data.inputs_np())
        is_pred_correct = test_probs.argmax(-1) == test_data.targets_np()

        if output_accuracies_grouped_by_celltype:
            df_by_celltype = pandas.DataFrame(
                {
                    "celltype": test_data.dataframe["celltype"],
                    "is_pred_correct": is_pred_correct,
                }
            )
            scores_sorted = (
                df_by_celltype.groupby("celltype")
                .agg(count=("is_pred_correct", "size"), accuracy=("is_pred_correct", "sum"))
                .sort_values(by="celltype", ascending=True)
            )
            with open(results_path.replace(".csv", ".celltype_scores.csv"), "a") as f:
                f.write(scores_sorted.to_csv(header=False))

        accuracy_traces = model.score(X=test_data.inputs_np(), y=test_data.targets_np())
        class_to_model_class = {
            original_class: model_class for model_class, original_class in enumerate(model.classes_)
        }
        remapped_targets_np = np.array([class_to_model_class[x] for x in test_data.targets_np()])
        try:
            accuracy_fields, _, _ = predictions_summed_over_unique_ids(test_probs, remapped_targets_np, field_ids)
        except ValueError:
            accuracy_fields = -1.0
        try:
            accuracy_retinas, _, _ = predictions_summed_over_unique_ids(test_probs, remapped_targets_np, retina_ids)
        except ValueError:
            accuracy_retinas = -1.0
        print(f"{accuracy_traces=:.1%} {accuracy_fields=:.1%} {accuracy_retinas=:.1%}")
        results_array = [accuracy_traces, accuracy_fields, accuracy_retinas]

        if do_plot_feature_importances:
            fig = plot_importances(model)
            fig.suptitle(f"Feature importances ({seed=} {accuracy_traces=:.1%} {accuracy_fields=:.1%}")
            fig.tight_layout()
            fig.savefig(results_path.replace(".csv", f"seed_{seed}_feature_importances_only_0.png"))

    else:
        predictions = model.predict(test_data.inputs_np())
        error_per_example = np.abs(predictions - test_data.targets_np())
        mean_error = np.average(error_per_example)
        mean_squared_error = np.sum(error_per_example**2) / predictions.shape[0]
        try:
            mean_error_fields, mean_squared_error_fields, _, _ = predictions_averaged_over_unique_ids(
                predictions, test_data.targets_np(), test_data.get_field_ids()
            )
        except ValueError:
            mean_error_fields = -1.0
            mean_squared_error_fields = -1.0
        try:
            mean_error_retina, mean_squared_error_retinas, _, _ = predictions_averaged_over_unique_ids(
                predictions, test_data.targets_np(), test_data.get_retina_ids()
            )
        except ValueError:
            mean_error_retina = -1.0
            mean_squared_error_retinas = -1.0
        print(f"{mean_error=:.1f} {mean_squared_error=:.1f}")
        print(f"{mean_error_fields=:.1f} {mean_squared_error_fields=:.1f}")
        print(f"{mean_error_retina=:.1f} {mean_squared_error_retinas=:.1f}")
        results_array = [mean_error, mean_error_fields, mean_error_retina]

    if plot_traces:
        experimenters_to_plot = test_data.LABELS
    else:
        experimenters_to_plot = []
    for i, experimenter in enumerate(experimenters_to_plot):
        if i in train_data.targets_np() and i in test_data.targets_np():
            probs_exp = test_probs[:, i]
            highest_idc = np.argpartition(probs_exp, -10)[-10:]
            chirps = test_data.dataframe["gchirp_average_norm"].iloc[highest_idc]
            avg_chirp = np.average([c for c in chirps], axis=0)
            fig, ax = plt.subplots(figsize=(16, 8))
            for ch_id, ch in enumerate(chirps):
                ax.plot(ch, label=f"Trace {ch_id}", alpha=0.5)
            ax.plot(avg_chirp, label="Average", linestyle="--", color="black")
            fig.suptitle(f"Ten most confident predictions for {experimenter}")
            fig.savefig(f"plots/{experimenter}_seed{seed}.png", dpi=300)
        else:
            print(f"Skipping {experimenter} as this experimenter isn't part of either the train or test set.")

    with open(results_path, "a") as f:
        line = ",".join(map(str, [seed, len(train_data), len(test_data)] + results_array))
        f.write(line + "\n")

    if do_save_confusion_matrix and train_data.should_use_classification():
        y_pred_ids = model.predict(test_data.inputs_np())
        postfix = f"random_forest_{seed}"
        save_confusion_matrix(test_data, y_pred_ids, dataset_name, postfix)
        y_pred_probs = model.predict_proba(test_data.inputs_np())
        save_prediction_probs(y_pred_probs, dataset_name, postfix)

    if export_random_forest_vis:
        from subprocess import call

        from sklearn.tree import export_graphviz

        # Export as dot file
        dot_fp = results_path.replace(".csv", f"_seed_{seed}.dot").replace("results/", "random_forest_vis/")
        png_fp = dot_fp.replace(".dot", ".png")
        export_graphviz(
            model.estimators_[0],
            out_file=dot_fp,
            class_names=["Cai", "Szatko"],
            rounded=True,
            proportion=False,
            precision=2,
            filled=True,
        )
        call(["dot", "-Tpng", dot_fp, "-o", png_fp, "-Gdpi=600"])


def main():
    args = parse_args()
    print(args)
    train_data, test_data = get_dataset_split_from_args(args)
    results_path = get_results_file_path(
        args.prediction,
        "random_forest",
        args.pca,
        args.fft,
        args.experimenters,
        args.location,
        args.celltype,
        args.lowpass_filter,
        args.features,
        args.do_not_quality_filter,
        field=args.field,
        supergroup=args.supergroup,
    )
    train(train_data, test_data, results_path, args.seed, dataset_name=args.prediction)


if __name__ == "__main__":
    main()
