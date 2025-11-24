import pickle
import random
import re
from collections import Counter
from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import pandas

from all_gcl_manuscript.batch_effects.datasets import NAME_TO_DATASET


def load_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def filter_data(
    data,
    gchirp_qi_threshold: float,
    bar_qi_threshold: float,
    control_values: Iterable[str] = ("control", "c1", "C1"),
    location: str | None = None,
    celltype: int | None = None,
    field: int | None = None,
    setup_id: str | None = "1",
    supergroup: str | None = None,
):
    data = data[data["cond1"].isin(tuple(control_values))]
    # Filter for chirp/bar for quality index
    data = data[(data["chirp_qidx"] >= gchirp_qi_threshold) | (data["bar_qidx"] >= bar_qi_threshold)]
    # Only use setup id 1 to not learn biases of different setups
    if setup_id:
        data = data[data["setupid"] == setup_id]
    if celltype is not None:
        data = data[data["group_id"] == celltype]
    if field is not None:
        regex = rf".*[^0-9]{field}[^0-9]*$"
        data = data[data["field"].str.match(regex)]
    if supergroup is not None:
        data = data[data["supergroup"] == supergroup]

    data = filter_location(data, location)
    data.reset_index()
    return data


def filter_location(data, location: str | None):
    if location is None:
        return data

    if location == "temporal_ventral":
        data = data[(data["ventral_dorsal_pos_um"] < 0) & (data["temporal_nasal_pos_um"] < 0)]
    elif location == "temporal_dorsal":
        data = data[(data["ventral_dorsal_pos_um"] > 0) & (data["temporal_nasal_pos_um"] < 0)]
    elif location == "nasal_ventral":
        data = data[(data["ventral_dorsal_pos_um"] < 0) & (data["temporal_nasal_pos_um"] > 0)]
    elif location == "nasal_dorsal":
        data = data[(data["ventral_dorsal_pos_um"] > 0) & (data["temporal_nasal_pos_um"] > 0)]
    elif location == "best":
        data_array = [
            data[(data["ventral_dorsal_pos_um"] < 0) & (data["temporal_nasal_pos_um"] < 0)],
            data[(data["ventral_dorsal_pos_um"] > 0) & (data["temporal_nasal_pos_um"] < 0)],
            data[(data["ventral_dorsal_pos_um"] < 0) & (data["temporal_nasal_pos_um"] > 0)],
            data[(data["ventral_dorsal_pos_um"] > 0) & (data["temporal_nasal_pos_um"] > 0)],
        ]
        # Select quadrant in which the experimenter with the lower number (trailing min())
        # of traces has the most traces (initial max)
        data = max(data_array, key=lambda x: x["experimenter"].value_counts().min())
    else:
        raise ValueError(f"Invalid location: {location}")

    return data


def get_dates_with_more_than_x_elements(
    data: pandas.core.frame.DataFrame,
    dates: list,
    min_elements: int | float,
) -> list:
    subset_dates, num_subset = [], 0
    for date in dates:
        num_subset += (data["date"] == date).sum()
        subset_dates.append(date)
        if num_subset >= min_elements:
            break
    return subset_dates


def get_train_test_split(
    data,
    seed: int,
    min_proportion_test: float = 0.2,
) -> tuple[list, list]:
    num_total = len(data)
    date_list = sorted(set(data["date"]))
    random.Random(seed).shuffle(date_list)

    test_dates = get_dates_with_more_than_x_elements(data, date_list, min_proportion_test * num_total)
    train_dates = date_list[len(test_dates) :]
    return train_dates, test_dates


def get_splits(
    data,
    seed: int = 44,
):
    train_dates, test_dates = get_train_test_split(data, seed, min_proportion_test=0.2)
    print(f"{len(train_dates)=}, {len(test_dates)=}")

    train_data = data[data["date"].isin(train_dates)]
    test_data = data[data["date"].isin(test_dates)]
    print(f"{len(train_data)} train, and {len(test_data)} test samples, total: {len(data)}")
    return train_data, test_data


def ensure_same_number_of_items(df, attribute: str = "experimenter"):
    min_rows = df[attribute].value_counts().min()

    def _filter_rows(x):
        return x.head(min_rows)

    df_filtered = df.groupby(attribute).apply(_filter_rows).reset_index(drop=True)
    return df_filtered


def get_equal_splits_of_two_attribute(
    data,
    seed: int,
    attribute_name: str = "experimenter",
    min_proportion_test: float = 0.2,
):
    # filter dataset by experimenter
    attribute_values = set(data[attribute_name])
    all_dates = sorted(set(data["date"]))
    random.Random(seed).shuffle(all_dates)

    min_vals_test = Counter(
        {a: int(len(data[data[attribute_name] == a]) * min_proportion_test) for a in attribute_values}
    )

    test_dates = []
    for date in all_dates:
        data_date = data[data["date"] == date]
        cnt = Counter(data_date[attribute_name])

        # check if there are any values that are still needed for test set
        if any(min_vals_test[val] > 0 for val in cnt.keys()):
            test_dates.append(date)
            min_vals_test -= cnt
            if len(min_vals_test) == 0:
                break

    train_dates = [d for d in all_dates if d not in test_dates]

    # make sure each experimenter has the same number of traces in each set
    train_data = data[data["date"].isin(train_dates)].sample(frac=1, random_state=seed)
    test_data = data[data["date"].isin(test_dates)].sample(frac=1, random_state=seed)
    print("Ensure same number of items for experimenter")
    train_data = ensure_same_number_of_items(train_data, attribute_name)
    test_data = ensure_same_number_of_items(test_data, attribute_name)
    assert len(set(train_data["date"]).intersection(set(test_data["date"]))) == 0, (
        "Train and test data have overlapping dates"
    )

    print(f"{len(train_data)} train, and {len(test_data)} test samples")
    return train_data, test_data


def create_ventral_dorsal_binary(
    data: pandas.DataFrame, attribute_name: str = "ventral_dorsal_binary"
) -> pandas.DataFrame:
    print("Adding ventralDorsalBinary, TODO: Check")
    data = data[data["ventral_dorsal_pos_um"].notna()]
    unknown_zone_min = -100
    unknown_zone_max = 100
    data = data[(data["ventral_dorsal_pos_um"] < unknown_zone_min) | (data["ventral_dorsal_pos_um"] > unknown_zone_max)]
    data[attribute_name] = (data["ventral_dorsal_pos_um"] >= unknown_zone_max).astype(int)
    return data


def create_sex_binary(data: pandas.DataFrame, attribute_name: str = "animgender") -> pandas.DataFrame:
    data = data[data["animgender"] != ""]
    assert len(set(data["animgender"])) == 2, f"Expected 2 animgenders, but {set(data['animgender'])=}"
    if attribute_name != "animgender":
        data[attribute_name] = (data["animgender"] == "male").astype(int)
    return data


def create_age_binary(
    data: pandas.DataFrame, young_age_weeks: int = 10, old_age_weeks: int = 15, attribute_name: str = "age_binary"
) -> pandas.DataFrame:
    # Calculate the time (index) until the absolute value of preproc_bar reaches each threshold from 0.1 to 1.0 in 0.1 steps
    def time_until_threshold(arr, threshold):
        abs_arr = np.abs(arr)
        threshold_value = threshold * np.max(abs_arr)
        above_thresh = np.where(abs_arr >= threshold_value)[0]
        res = float(above_thresh[0]) if len(above_thresh) > 0 else np.nan
        return res

    data["preproc_bar_time_until_0.5"] = data["preproc_bar"].apply(lambda x: time_until_threshold(x, 0.5))
    # add 8th value of preproc_bar as it had a slightly higher importance for the random forest
    for idx in range(data["preproc_bar"].iloc[0].shape[0]):
        data[f"preproc_bar_{idx}"] = data["preproc_bar"].apply(lambda x: x[idx])
        data[f"abs_preproc_bar_{idx}"] = data[f"preproc_bar_{idx}"].abs()
    # filter out inplausible values
    data = data[(data["age"].notna()) & (data["age"] > 5) & (data["age"] < 200)]
    # filter selected range
    data = data[(data["age"] <= young_age_weeks * 7) | (data["age"] >= old_age_weeks * 7)]
    data[attribute_name] = (data["age"] >= old_age_weeks * 7).astype(int)
    return data


def create_field_id_binary(
    data: pandas.DataFrame, field_id_a: int = 0, field_id_b: int = 1, attribute_name: str = "field_id_binary"
) -> pandas.DataFrame:
    data = data[data["field"].notna()]
    data = data[data["setupid"] == "1"]
    print("Filtered for setupid 1")

    # build a map of dates to field ids
    def field_id(value):
        return int(re.search(r"\d+", value).group(0))

    field_ids_map_for_date_exp_num: dict[tuple[str, int], dict[str, int]] = {}
    for date in set(data["date"]):
        data_date = data[data["date"] == date]
        for exp_num in set(data_date["exp_num"]):
            data_exp_num = data_date[data_date["exp_num"] == exp_num]
            fields = sorted(set(data_exp_num["field"]))
            field_ids = [field_id(f) for f in fields]
            if set(range(min(field_ids), max(field_ids) + 1)) != set(field_ids):
                fields = sorted(set(data_exp_num["field"]))
                print(
                    f"Field ids ({fields}) are not consecutive for {date=} {exp_num=}"
                    f"experimenter: {set(data_exp_num['experimenter'])}"
                )
            elif min(field_ids) > 1:
                print(
                    f"Field ids ({fields}) start at {min(field_ids)} for {date=} {exp_num=} "
                    f"experimenter: {set(data_exp_num['experimenter'])}"
                )
            elif any(f.endswith("a") or f.endswith("b") for f in fields):
                # ignore fields for which retina was cut in two parts
                pass
            else:
                field_id_map = {
                    field: field_id - min(field_ids) for field, field_id in zip(fields, field_ids, strict=True)
                }
                assert min(field_id_map.values()) == 0
                assert max(field_id_map.values()) == len(field_ids) - 1
                field_ids_map_for_date_exp_num[(date, exp_num)] = field_id_map

    def field_id_corrected(row):
        date, exp_num = row["date"], row["exp_num"]
        field_id_map = field_ids_map_for_date_exp_num.get((date, exp_num))
        if field_id_map is not None:
            return field_id_map[row["field"]]
        else:
            return -1

    data["field_id"] = data.apply(axis=1, func=field_id_corrected)  # type: ignore
    data = data[data["field_id"].isin({field_id_a, field_id_b})]
    data[attribute_name] = (data["field_id"] == field_id_b).astype(bool)
    return data


def get_dataset_splits(
    file_path: str,
    dataset_name: str,
    experimenters: list[str] | None,
    features: list[str],
    pca: bool,
    fft: bool,
    seed: int,
    location: str | None,
    celltype: int | None,
    lowpass_filter: float | None,
    do_not_quality_filter: bool,
    field: int | None,
    supergroup: str | None,
) -> tuple[Any, Any]:
    data = load_data(file_path)

    # adapt dataframe to binary ventral dorsal split
    if dataset_name == "ventralDorsalBinary":
        data = create_ventral_dorsal_binary(data)
    elif dataset_name == "sex":
        data = data[data["animgender"] != ""].reset_index(drop=True)
    elif dataset_name.startswith("fieldIdBinary"):
        try:
            dataset_name, field_id_a_str, field_id_b_str = dataset_name.split("-")
            field_id_a, field_id_b = int(field_id_a_str), int(field_id_b_str)
        except:
            field_id_a, field_id_b = 0, 1
        print(f"{dataset_name=}, {field_id_a=}, {field_id_b=}")
        data = create_field_id_binary(data, field_id_a, field_id_b)
    elif dataset_name == "ageBinary":
        data = create_age_binary(data)
    if do_not_quality_filter:
        chirp_threshold, bar_threshold = 0.0, 0.0
    else:
        chirp_threshold, bar_threshold = 0.35, 0.6
    setup_id = None if dataset_name == "setupid" else "1"
    data = filter_data(
        data, chirp_threshold, bar_threshold, location=location, celltype=celltype, field=field, setup_id=setup_id,
        supergroup=supergroup,
    )
    # filter for experimenters
    if experimenters is not None:
        data = data[data["experimenter"].isin(experimenters)]

    dataset_class = NAME_TO_DATASET[dataset_name]
    dataset_cl = partial(
        dataset_class,
        input_names=features,
        do_apply_pca=pca,  # type: ignore
        use_fft_features=fft,
        low_pass_filter_frequency=lowpass_filter,
    )

    datasets_to_balance_map = {
        "sex": "animgender",
        "setupid": "setupid",
        "ventralDorsalBinary": "ventral_dorsal_binary",
        "fieldIdBinary": "field_id_binary",
        "ageBinary": "age_binary",
    }
    if experimenters is not None and len(experimenters) == 2:
        print("Splitting equally between two experimenters")
        train_data, test_data = map(
            dataset_cl, get_equal_splits_of_two_attribute(data, seed=seed, attribute_name="experimenter")
        )
    elif dataset_name in datasets_to_balance_map:
        attribute_name = datasets_to_balance_map[dataset_name]
        print(f"Balancing attribute {attribute_name} for dataset {dataset_name}.")
        train_data, test_data = map(
            dataset_cl, get_equal_splits_of_two_attribute(data, seed=seed, attribute_name=attribute_name)
        )
    else:
        print(f"Not balancing {dataset_name}")
        dataset_cl = partial(
            dataset_class,
            input_names=features,
            do_apply_pca=pca,  # type: ignore
            use_fft_features=fft,
            low_pass_filter_frequency=lowpass_filter,
        )
        train_data, test_data = map(dataset_cl, get_splits(data, seed=seed))

    intersection_dates = set(train_data.dataframe["date"]).intersection(set(test_data.dataframe["date"]))
    assert len(intersection_dates) == 0, "Train and test data have overlapping dates"
 
    if len(train_data) == 0 or len(test_data) == 0 or train_data.num_unique_targets() <= 1 or test_data.num_unique_targets() <= 1:
        raise ValueError(f"train or test data is empty or only have one unique target: {train_data=} {test_data=}")
    return train_data, test_data


def get_dataset_split_from_args(args) -> tuple[Any, Any]:
    return get_dataset_splits(
        file_path=args.data_frame_path,
        dataset_name=args.prediction,
        experimenters=args.experimenters,
        features=args.features,
        pca=args.pca,
        fft=args.fft,
        seed=args.seed,
        location=args.location,
        celltype=args.celltype,
        lowpass_filter=args.lowpass_filter,
        do_not_quality_filter=args.do_not_quality_filter,
        field=args.field,
        supergroup=args.supergroup
    )
