import datajoint as dj
import numpy as np

from djimaging.tables import core, misc, receptivefield, location, response, spike_estimation, classifier_v2
from djimaging.user.all_rgcs import tables as my_tables

schema = dj.Schema()


@schema
class UserInfo(core.UserInfoTemplate):
    pass


@schema
class RawDataParams(core.RawDataParamsTemplate):
    userinfo_table = UserInfo


@schema
class Experiment(core.ExperimentTemplate):
    userinfo_table = UserInfo

    class ExpInfo(core.ExperimentTemplate.ExpInfo):
        pass

    class Animal(core.ExperimentTemplate.Animal):
        pass

    class Indicator(core.ExperimentTemplate.Indicator):
        pass

    class PharmInfo(core.ExperimentTemplate.PharmInfo):
        pass


@schema
class AnimalAge(misc.AnimalAgeTemplate):
    experiment_table = Experiment


@schema
class Field(core.FieldTemplate):
    incl_region = True  # Include region as primary key?
    incl_cond1 = False  # Include condition 1 as primary key?
    incl_cond2 = False  # Include condition 2 as primary key?
    incl_cond3 = False  # Include condition 3 as primary key?

    userinfo_table = UserInfo
    raw_params_table = RawDataParams
    experiment_table = Experiment

    class StackAverages(core.FieldTemplate.StackAverages):
        pass


@schema
class Stimulus(core.StimulusTemplate):
    pass


@schema
class Presentation(core.PresentationTemplate):
    incl_region = True  # Include region as primary key?
    incl_cond1 = True  # Include condition 1 as primary key?
    incl_cond2 = False  # Include condition 2 as primary key?
    incl_cond3 = False  # Include condition 3 as primary key?

    userinfo_table = UserInfo
    experiment_table = Experiment
    field_table = Field
    stimulus_table = Stimulus
    raw_params_table = RawDataParams

    class ScanInfo(core.PresentationTemplate.ScanInfo):
        pass

    class StackAverages(core.PresentationTemplate.StackAverages):
        pass


# Misc
@schema
class HighRes(misc.HighResTemplate):
    incl_region = True  # Include region as primary key?
    incl_cond1 = False  # Include condition 1 as primary key?
    incl_cond2 = False  # Include condition 2 as primary key?
    incl_cond3 = False  # Include condition 3 as primary key?
    _fallback_to_raw = True

    field_table = Field
    experiment_table = Experiment
    userinfo_table = UserInfo
    raw_params_table = RawDataParams

    class StackAverages(misc.HighResTemplate.StackAverages):
        pass


@schema
class DataRoiMask(my_tables.RoiMaskDataTemplate):
    field_table = Field
    raw_params_table = RawDataParams
    presentation_table = Presentation
    userinfo_table = UserInfo

    class RoiMaskPresentation(my_tables.RoiMaskDataTemplate.RoiMaskPresentation):
        presentation_table = Presentation


@schema
class Roi(core.RoiTemplate):
    roi_mask_table = DataRoiMask
    userinfo_table = UserInfo
    field_table = Field


@schema
class Traces(core.TracesTemplate):
    userinfo_table = UserInfo
    raw_params_table = RawDataParams
    presentation_table = Presentation
    roi_table = Roi
    roi_mask_table = DataRoiMask


@schema
class PreprocessParams(core.PreprocessParamsTemplate):
    stimulus_table = Stimulus


@schema
class PreprocessTraces(core.PreprocessTracesTemplate):
    _baseline_max_dt = 2.  # seconds before stimulus used for baseline calculation

    presentation_table = Presentation
    preprocessparams_table = PreprocessParams
    traces_table = Traces


@schema
class Snippets(core.SnippetsTemplate):
    _pad_trace = True
    _dt_base_line_dict = {
        'chirp': 8 * 0.128,
        'gChirp': 8 * 0.128,
        'lChirp': 8 * 0.128,
        'movingbar': 5 * 0.128,
        'mbar': 5 * 0.128,
    }

    stimulus_table = Stimulus
    presentation_table = Presentation
    traces_table = Traces
    preprocesstraces_table = PreprocessTraces


@schema
class Averages(core.ResampledAveragesTemplate):
    _f_resample = 60  # Frequency in Hz to resample averages
    _norm_kind = 'amp_one'  # How to normalize averages?

    snippets_table = Snippets


@schema
class Averages78125Hz(core.ResampledAveragesTemplate):
    _f_resample = 7.8125  # Frequency in Hz to resample averages
    _norm_kind = 'amp_one'  # How to normalize averages?

    snippets_table = Snippets


@schema
class OpticDisk(location.OpticDiskTemplate):
    incl_region = True
    field_table = Field

    userinfo_table = UserInfo
    experiment_table = Experiment
    raw_params_table = RawDataParams


@schema
class RelativeFieldLocation(location.RelativeFieldLocationTemplate):
    field_table = Field
    opticdisk_table = OpticDisk


@schema
class RetinalFieldLocation(location.RetinalFieldLocationTemplate):
    relativefieldlocation_table = RelativeFieldLocation
    expinfo_table = Experiment.ExpInfo


@schema
class RelativeRoiLocationWrtField(location.RelativeRoiLocationWrtFieldTemplate):
    roi_table = Roi
    roi_mask_table = DataRoiMask
    field_table = Field
    presentation_table = Presentation


@schema
class RelativeRoiLocation(location.RelativeRoiLocationTemplate):
    relative_field_location_wrt_field_table = RelativeRoiLocationWrtField
    relative_field_location_table = RelativeFieldLocation
    field_table = Field


@schema
class RetinalRoiLocation(location.RetinalRoiLocationTemplate):
    relative_roi_location_table = RelativeRoiLocation
    expinfo_table = Experiment.ExpInfo


@schema
class ChirpQI(response.ChirpQITemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets


@schema
class ChirpFeatures(response.ChirpFeaturesRgcTemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets
    presentation_table = Presentation


@schema
class OsDsIndexes(response.OsDsIndexesTemplate):
    _reduced_storage = True
    _n_shuffles = 1000
    _version = 2

    stimulus_table = Stimulus
    snippets_table = Snippets


# Classification
@schema
class Baden16TracesV2(classifier_v2.Baden16TracesV2Template):
    _stim_name_chirp = 'chirp'
    _stim_name_bar = 'movingbar'

    traces_table = Traces
    presentation_table = Presentation
    stimulus_table = Stimulus


@schema
class ClassifierV2(classifier_v2.ClassifierV2Template):
    pass


@schema
class CelltypeAssignmentV2(classifier_v2.CelltypeAssignmentV2Template):
    classifier_table = ClassifierV2
    baden_trace_table = Baden16TracesV2
    field_table = Field
    roi_table = Roi


@schema
class CascadeTraceParams(spike_estimation.CascadeTracesParamsTemplate):
    stimulus_table = Stimulus


@schema
class CascadeTraces(spike_estimation.CascadeTracesTemplate):
    cascadetraces_params_table = CascadeTraceParams
    presentation_table = Presentation
    traces_table = Traces


@schema
class CascadeParams(spike_estimation.CascadeParamsTemplate):
    pass


@schema
class CascadeSpikes(spike_estimation.CascadeSpikesTemplate):
    presentation_table = Presentation
    cascadetraces_params_table = CascadeTraceParams
    cascadetraces_table = CascadeTraces
    cascade_params_table = CascadeParams


# Misc
@schema
class LightArtifact(misc.LightArtifactTemplate):
    userinfo_table = UserInfo
    presentation_table = Presentation
    raw_params_table = RawDataParams
    stimulus_table = Stimulus


# Quality Index for Presentations
@schema
class ChirpQIPresentation(response.RepeatQIPresentationTemplate):
    _qidx_col = 'qidx'
    _levels = (0.25, 0.3, 0.35, 0.4, 0.45)
    presentation_table = Presentation
    qi_table = ChirpQI


@schema
class BarQIPresentation(response.RepeatQIPresentationTemplate):
    _qidx_col = 'd_qi'
    _levels = (0.5, 0.6, 0.7)
    presentation_table = Presentation
    qi_table = OsDsIndexes


# STA
@schema
class FastStaParams(receptivefield.FastStaParamsTemplate):
    stimulus_table = Stimulus
    presentation_table = Presentation


@schema
class FastSta(receptivefield.FastStaTemplate):
    params_table = FastStaParams
    presentation_table = Presentation
    traces_table = PreprocessTraces


@schema
class SplitRFParams(receptivefield.SplitRFParamsTemplate):
    _max_dt_future = np.inf


@schema
class SplitRF(receptivefield.SplitRFTemplate):
    rf_table = FastSta
    split_rf_params_table = SplitRFParams


@schema
class FitGauss2DRF(receptivefield.FitGauss2DRFTemplate):
    split_rf_table = SplitRF
    stimulus_table = Stimulus


@schema
class TempRFProperties(receptivefield.TempRFPropertiesTemplate):
    _max_dt_future = np.inf
    split_rf_table = SplitRF
    rf_table = FastSta


@schema
class RfOffset(receptivefield.RfOffsetTemplate):
    stimulus_tab = Stimulus
    rf_split_tab = SplitRF
    rf_fit_tab = FitGauss2DRF


@schema
class RfRoiOffset(receptivefield.RfRoiOffsetTemplate):
    rf_offset_tab = RfOffset
    roi_pos_wrt_field_tab = RelativeRoiLocationWrtField
    userinfo_tab = UserInfo
    experiment_tab = Experiment
    stimulus_tab = Stimulus
    pres_tab = Presentation
    roimask_tab = DataRoiMask.RoiMaskPresentation
    rf_fit_tab = FitGauss2DRF

    def make(self, key):
        rf_dx_um, rf_dy_um = (self.rf_offset_tab & key).fetch1('rf_dx_um', 'rf_dy_um')
        relx_wrt_field, rely_wrt_field = (self.roi_pos_wrt_field_tab & key).fetch1(
            'relx_wrt_field', 'rely_wrt_field')

        setupid = (self.experiment_tab.ExpInfo & key).fetch1('setupid')

        # If you stand in front of the setup:
        # -rely is this axis: → (left to right)
        # +relx is this axis: ↑ (front to back, typically ventral to dorsal)

        # For the RF x and y can have different meaning,
        # dependent in which direction the dense noise stimulus was provided by the user.

        if str(setupid) == "1":
            rely_rf_roi_um = - rf_dx_um - rely_wrt_field  # RF x is aligned with -rely axis
            relx_rf_roi_um = + rf_dy_um - relx_wrt_field  # RF y is aligned with relx axis
        elif str(setupid) == "3":
            rely_rf_roi_um = - rf_dx_um - rely_wrt_field  # RF x is aligned with -rely axis
            relx_rf_roi_um = - rf_dy_um + relx_wrt_field  # RF y is aligned with relx axis
        else:
            raise NotImplementedError

        self.insert1(dict(**key, relx_rf_roi_um=relx_rf_roi_um, rely_rf_roi_um=rely_rf_roi_um))
