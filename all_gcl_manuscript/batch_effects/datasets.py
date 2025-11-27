from abc import abstractmethod
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache

import numpy as np
import scipy
import torch


def downsample(signal: np.ndarray, dt_old: float, dt_new: float) -> np.ndarray:
    time_old = np.arange(signal.size) * dt_old
    factor = dt_old / dt_new
    time_new = np.arange(int(signal.size * factor)) * dt_new
    signal_new = scipy.interpolate.interp1d(
        time_old, signal, assume_sorted=True, bounds_error=False, fill_value="extrapolate"
    )(time_new)
    return signal_new


@lru_cache(2)
def create_low_pass_filter(cutoff: float, fs: float, order: int = 5) -> tuple[np.ndarray, np.ndarray]:
    return scipy.signal.butter(order, cutoff, fs=fs, btype="lowpass", analog=False, output="ba")  # type: ignore


def low_pass_filter(data, cutoff: float, fs: float, order: int = 5):
    b, a = create_low_pass_filter(cutoff=cutoff, fs=fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y


def compute_fft_amplitudes(signal: np.ndarray, frequency: float) -> tuple[np.ndarray, np.ndarray]:
    # Compute the FFT
    n = len(signal)
    fhat = np.fft.fft(signal, n)  # compute FFT
    PSD = np.abs(fhat) / n  # power spectrum (power per freq)
    freq = (frequency / n) * np.arange(n)  # create x-axis of frequencies in Hz
    half_idx = int(np.floor(n / 2))
    return freq[:half_idx], PSD[:half_idx]  # only return first half of freqs


class BaseDataset(torch.utils.data.Dataset):
    DEFAULT_FREQUENCY = 7.81
    DEFAULT_DT = 1.0 / DEFAULT_FREQUENCY
    FEATURES_NAME_TO_DT_NAME = {
        "chirp_60Hz_average_norm": "chirp_60Hz_average_dt",
        "chirp_8Hz_average_norm": "chirp_8Hz_average_dt",
    }

    def __init__(
        self,
        dataframe,
        input_names: Iterable[str],
        targets: torch.Tensor,
        use_fft_features: bool = False,
        low_pass_filter_frequency: float | None = None,
    ):
        super().__init__()
        if len(dataframe) == 0:
            raise ValueError("Warning: dataframe has no items")
        input_features = []
        for name in input_names:
            if "-" in name:
                name, start_idx_str, end_idx_str = name.split("-")
                start_idx, end_idx = int(start_idx_str), int(end_idx_str)
            else:
                start_idx, end_idx = None, None
            if name in self.FEATURES_NAME_TO_DT_NAME:
                dt_set = set(dataframe[self.FEATURES_NAME_TO_DT_NAME[name]])
                assert len(dt_set) == 1
                dt = list(dt_set)[0]
                assert start_idx is None and end_idx is None
                feat = np.stack([downsample(x, dt, self.DEFAULT_DT) for x in dataframe[name]])
            else:
                feat = np.stack([np.array([x]) if type(x) is float else x[start_idx:end_idx] for x in dataframe[name]])  # type: ignore
                if len(feat.shape) == 1:
                    feat = np.expand_dims(feat, 1)
            if low_pass_filter_frequency is not None:
                feat_ar = [
                    low_pass_filter(x, cutoff=low_pass_filter_frequency, fs=self.DEFAULT_FREQUENCY) for x in feat
                ]
                feat = np.stack(feat_ar)
            if use_fft_features:
                feat = np.stack([compute_fft_amplitudes(x, self.DEFAULT_FREQUENCY)[1] for x in feat])
            input_features.append(feat)
        inputs_np = np.concatenate(input_features, axis=-1)
        self._input_names = input_names
        self.inputs = torch.FloatTensor(inputs_np).unsqueeze(-1)
        self.targets = targets
        self.dataframe = dataframe

    def inputs_np(self) -> np.ndarray:
        return self.inputs.detach().numpy()[..., 0]

    def targets_np(self) -> np.ndarray:
        return self.targets.detach().numpy()

    def field(self) -> list[str]:
        return [x for x in self.dataframe["field"]]

    def num_unique_targets(self) -> int:
        return len(np.unique(self.targets))

    def date(self) -> list[str]:
        return [x for x in self.dataframe["date"]]

    def exp_num(self) -> list[int]:
        return [int(x) for x in self.dataframe["exp_num"]]

    def experimenter(self) -> list[str]:
        return [x for x in self.dataframe["experimenter"]]

    def get_field_ids(self) -> list[str]:
        return [
            "_".join(map(str, x))
            for x in zip(self.experimenter(), self.date(), self.exp_num(), self.field(), strict=True)
        ]

    def get_retina_ids(self) -> list[str]:
        return ["_".join(map(str, x)) for x in zip(self.experimenter(), self.date(), self.exp_num(), strict=True)]

    @abstractmethod
    def should_use_classification(self) -> bool:
        raise NotImplementedError()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]

    def __str__(self):
        if self.should_use_classification():
            counter = Counter([int(x) for x in self.targets_np()])
            return f"{self.__class__.__name__}: {len(self)} Examples, features={self._input_names}, targets={counter}"
        else:
            return (
                f"{self.__class__.__name__}: {len(self)} Examples, features={self._input_names}, "
                f"mean={self.bias_init_value:.1f}"
            )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.inputs.shape=}, {self.targets.shape=})"


class GenderDataset(BaseDataset):
    LABELS = ["female", "male"]
    ID_TO_LABEL = dict(enumerate(LABELS))
    LABEL_TO_ID = {e: i for i, e in ID_TO_LABEL.items()}

    def __init__(
        self,
        dataframe,
        input_names: Iterable[str],
        use_fft_features: bool = False,
        low_pass_filter_frequency: float | None = None,
    ):
        data = dataframe[dataframe["animgender"].isin(["male", "female"])]
        targets = torch.LongTensor([int(x == "female") for x in data["animgender"]])
        self.num_categories = len(self.LABELS)
        super().__init__(data, input_names, targets, use_fft_features, low_pass_filter_frequency)

    def should_use_classification(self) -> bool:
        return True


class RgcGroupDataset(BaseDataset):
    def __init__(
        self,
        dataframe,
        input_names: Iterable[str],
        use_fft_features: bool = False,
        low_pass_filter_frequency: float | None = None,
    ):
        targets = torch.LongTensor([x - 1 for x in dataframe["celltype"]])
        self.num_categories = len(set(dataframe["celltype"]))
        super().__init__(dataframe, input_names, targets, use_fft_features, low_pass_filter_frequency)

    def should_use_classification(self) -> bool:
        return True


class SetupIdDataset(BaseDataset):
    _SETUP_ID_KEY = "setupid"
    LABELS = ["1", "2", "3"]
    ID_TO_LABEL = dict(enumerate(LABELS))
    LABEL_TO_ID = {e: i for i, e in ID_TO_LABEL.items()}

    def __init__(
        self,
        dataframe,
        input_names: Iterable[str],
        use_fft_features: bool = False,
        low_pass_filter_frequency: float | None = None,
    ):
        targets = torch.LongTensor([self.LABEL_TO_ID[id_] for id_ in dataframe[self._SETUP_ID_KEY]])
        self.num_categories = len(self.LABELS)
        super().__init__(dataframe, input_names, targets, use_fft_features, low_pass_filter_frequency)

    def should_use_classification(self) -> bool:
        return True


class ExperimenterDataset(BaseDataset):
    LABELS = ["Arlinghaus", "Cai", "Debinski", "Dyszkant", "Franke", "Gonschorek", "Schwerd-Kleine", "Szatko"]
    ID_TO_LABEL = dict(enumerate(LABELS))
    LABEL_TO_ID = {e: i for i, e in ID_TO_LABEL.items()}

    def __init__(
        self,
        dataframe,
        input_names: Iterable[str],
        use_fft_features: bool = False,
        low_pass_filter_frequency: float | None = None,
    ):
        targets = torch.LongTensor([self.LABEL_TO_ID[x] for x in dataframe["experimenter"]])
        self.num_categories = len(self.LABELS)
        super().__init__(dataframe, input_names, targets, use_fft_features, low_pass_filter_frequency)

    def should_use_classification(self) -> bool:
        return True


class AgeDataset(BaseDataset):
    def __init__(
        self,
        dataframe,
        input_names: Iterable[str],
        use_fft_features: bool = False,
        low_pass_filter_frequency: float | None = None,
    ):
        age_greater_zero = [age > 0 for age in dataframe["age"]]
        dataframe = dataframe[age_greater_zero]
        targets = torch.FloatTensor([age for age in dataframe["age"]])
        self.bias_init_value = targets.mean().item()

        super().__init__(dataframe, input_names, targets, use_fft_features, low_pass_filter_frequency)

    def should_use_classification(self) -> bool:
        return False


class AgeBinaryDataset(BaseDataset):
    LABELS = ["young", "old"]

    def __init__(
        self,
        dataframe,
        input_names: Iterable[str],
        use_fft_features: bool = False,
        low_pass_filter_frequency: float | None = None,
    ):
        targets = torch.LongTensor(dataframe["age_binary"].values)

        super().__init__(dataframe, input_names, targets, use_fft_features, low_pass_filter_frequency)

    def should_use_classification(self) -> bool:
        return True


class TemporalNasalPositionDataset(BaseDataset):
    def __init__(
        self,
        dataframe,
        input_names: Iterable[str],
        use_fft_features: bool = False,
        low_pass_filter_frequency: float | None = None,
    ):
        dataframe = dataframe[dataframe["temporal_nasal_pos_um"].notna()]
        targets = torch.FloatTensor(dataframe["temporal_nasal_pos_um"].values)
        self.bias_init_value = targets.mean().item()

        super().__init__(dataframe, input_names, targets, use_fft_features, low_pass_filter_frequency)

    def should_use_classification(self) -> bool:
        return False


class FieldIdBinaryDataset(BaseDataset):
    LABELS = ["field_id_a", "field_id_b"]

    def __init__(
        self,
        dataframe,
        input_names: Iterable[str],
        use_fft_features: bool = False,
        low_pass_filter_frequency: float | None = None,
    ):
        targets = torch.LongTensor(dataframe["field_id_binary"].values)

        super().__init__(dataframe, input_names, targets, use_fft_features, low_pass_filter_frequency)

    def should_use_classification(self) -> bool:
        return True


class VentralDorsalPositionDataset(BaseDataset):
    def __init__(
        self,
        dataframe,
        input_names: Iterable[str],
        use_fft_features: bool = False,
        low_pass_filter_frequency: float | None = None,
    ):
        dataframe = dataframe[dataframe["ventral_dorsal_pos_um"].notna()]
        targets = torch.FloatTensor(dataframe["ventral_dorsal_pos_um"].values)
        self.bias_init_value = targets.mean().item()

        super().__init__(dataframe, input_names, targets, use_fft_features, low_pass_filter_frequency)

    def should_use_classification(self) -> bool:
        return False


class VentralDorsalBinaryDataset(BaseDataset):
    LABELS = ["ventral", "dorsal"]

    def __init__(
        self,
        dataframe,
        input_names: Iterable[str],
        use_fft_features: bool = False,
        low_pass_filter_frequency: float | None = None,
    ):
        assert "ventral_dorsal_binary" in dataframe
        targets = torch.LongTensor(dataframe["ventral_dorsal_binary"].values)
        super().__init__(dataframe, input_names, targets, use_fft_features, low_pass_filter_frequency)

    def should_use_classification(self) -> bool:
        return True


NAME_TO_DATASET = {
    "age": AgeDataset,
    "sex": GenderDataset,
    "experimenter": ExperimenterDataset,
    "rgcgroup": RgcGroupDataset,
    "temporalNasal": TemporalNasalPositionDataset,
    "ventralDorsal": VentralDorsalPositionDataset,
    "ventralDorsalBinary": VentralDorsalBinaryDataset,
    "fieldIdBinary": FieldIdBinaryDataset,
    "setupid": SetupIdDataset,
    "ageBinary": AgeBinaryDataset,
}
