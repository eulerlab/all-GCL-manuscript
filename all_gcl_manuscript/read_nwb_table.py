"""
Functions to read NWB files and reconstruct the comprehensive table.

This module provides functionality to read exported NWB files and reconstruct
the same comprehensive table that was used to run analysis in the all-GCL manuscript.
"""

import glob
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from tqdm.auto import tqdm

from .utils import serialize_numpy_arrays, strip_nan_trailing

logger = logging.getLogger(__name__)


def read_comprehensive_table_from_nwb(nwb_file_path: str) -> Optional[pd.DataFrame]:
    """Read and reconstruct the comprehensive table from modular data in an NWB file.

    This function reconstructs the same table structure as the one used to run analysis in the all-GCL manuscript by
    reading data from the various processing modules (spatial, condition-specific, etc.)
    and combining them into a unified table with one row per ROI per condition.

    Args:
        nwb_file_path: Path to the NWB file

    Returns:
        DataFrame containing the reconstructed comprehensive table, or None if not found
    """
    try:
        with NWBHDF5IO(nwb_file_path, "r") as io:
            nwbfile = io.read()

            # Check if ROI segmentation exists (required)
            if "rois" not in nwbfile.processing:
                logger.warning(f"No ROI segmentation found in {nwb_file_path}")
                return None

            img_seg = nwbfile.processing["rois"]["ImageSegmentation"]
            plane_segmentation = img_seg["ROI masks"]

            # Get basic ROI information
            # Use explicit ROI ids from the plane segmentation; they may be non-consecutive
            n_rois = len(plane_segmentation)
            try:
                roi_ids = [int(x) for x in plane_segmentation.id.data[:]]
            except Exception:
                # Fallback to sequential ids if .id dataset is unavailable
                # roi_ids = list(range(1, n_rois + 1))
                logger.error(f"No ROI ids found in the plane segmentation for file {nwb_file_path}", exc_info=True)

            assert len(roi_ids) == n_rois, (
                "Number of ROIs in the plane segmentation does not match the number of ROIs in the table: "
                f"{len(roi_ids)} != {n_rois}"
            )

            # Build mappings: segmentation row index (as stored) and time-series column index
            # Segmentation rows are stored in insertion order (export used roi_id-sorted order)
            roi_id_to_seg_index = {rid: idx for idx, rid in enumerate(roi_ids)}
            # Time-series arrays were stacked in ascending roi_id order during export
            roi_ids_sorted = roi_ids
            roi_id_to_ts_index = {rid: idx for idx, rid in enumerate(roi_ids_sorted)}

            # Get ROI sizes and other basic info from the segmentation table
            roi_data = {}
            if "roi_sizes_um2" in plane_segmentation.colnames:
                roi_sizes_um2 = plane_segmentation["roi_sizes_um2"].data[:]
                for roi_id in roi_ids:
                    seg_idx = roi_id_to_seg_index.get(roi_id)
                    if seg_idx is not None and seg_idx < len(roi_sizes_um2):
                        roi_data[roi_id] = {"roi_size_um2": roi_sizes_um2[seg_idx]}
            else:
                # If not available, set to NaN
                for roi_id in roi_ids:
                    roi_data[roi_id] = {"roi_size_um2": np.nan}

            # Add field dimensions if stored as segmentation columns
            if "nxpix" in plane_segmentation.colnames:
                nxpix_vals = plane_segmentation["nxpix"].data[:]
                for roi_id in roi_ids:
                    seg_idx = roi_id_to_seg_index.get(roi_id)
                    if seg_idx is not None and seg_idx < len(nxpix_vals):
                        roi_data.setdefault(roi_id, {})["nxpix"] = int(nxpix_vals[seg_idx])
            if "nypix" in plane_segmentation.colnames:
                nypix_vals = plane_segmentation["nypix"].data[:]
                for roi_id in roi_ids:
                    seg_idx = roi_id_to_seg_index.get(roi_id)
                    if seg_idx is not None and seg_idx < len(nypix_vals):
                        roi_data.setdefault(roi_id, {})["nypix"] = int(nypix_vals[seg_idx])

            # Extract session info from NWB metadata
            session_metadata = {
                "session_id": nwbfile.identifier,
                "session_description": nwbfile.session_description,
                "experimenter": nwbfile.experimenter[0] if nwbfile.experimenter else "unknown",
                "session_start_time": nwbfile.session_start_time,
            }

            # Parse session ID to extract experimenter and date (experimenter_date format)
            session_parts = session_metadata["session_id"].split("_")
            if len(session_parts) >= 2:
                experimenter = session_parts[0]
                date = session_parts[1]
            else:
                experimenter = session_metadata["experimenter"]
                date = session_metadata["session_start_time"].strftime("%Y-%m-%d")

            # Extract exp_num and region from the directory structure
            # Path structure: .../session_{experimenter}_{date}/experiment_{exp_num}_{region}/field_{field_id}.nwb
            path_parts = nwb_file_path.replace("\\", "/").split("/")
            exp_num = 1
            region = "unknown"

            # Find the experiment directory
            for part in path_parts:
                if part.startswith("experiment_"):
                    exp_parts = part.split("_")
                    if len(exp_parts) >= 3:  # experiment_{exp_num}_{region...}
                        try:
                            exp_num = int(exp_parts[1])
                        except ValueError:
                            exp_num = exp_parts[1]  # Keep as string if not numeric
                        region = "_".join(exp_parts[2:])  # Region might have underscores
                    break

            # Extract field info from file name
            file_name = os.path.basename(nwb_file_path)
            if "field_" in file_name:
                field = file_name.replace(".nwb", "").replace("field_", "")
            else:
                field = "unknown"

            # Get animal and experiment metadata
            experiment_metadata = {}
            if nwbfile.subject:
                experiment_metadata.update(
                    {
                        "genline": nwbfile.subject.genotype or "unknown",
                        "animgender": nwbfile.subject.sex or "unknown",
                        "age": nwbfile.subject.age or "unknown",
                    }
                )

            if hasattr(nwbfile, "lab_meta_data") and nwbfile.lab_meta_data:
                for meta_name in nwbfile.lab_meta_data:
                    meta_obj = nwbfile.lab_meta_data[meta_name]
                    if hasattr(meta_obj, "eye"):
                        experiment_metadata["eye"] = meta_obj.eye
                    if hasattr(meta_obj, "preparation_orientation"):
                        experiment_metadata["prepwmorient"] = meta_obj.preparation_orientation
                    if hasattr(meta_obj, "setup_id"):
                        experiment_metadata["setupid"] = meta_obj.setup_id

            # Add missing metadata columns to match original table
            experiment_metadata.update(
                {
                    "region": region,  # Default for these sessions
                    "classifier_id": 1,  # Not available in NWB
                }
            )

            # Get imaging plane info for field metadata
            field_metadata = {}
            if "ImagingPlane" in nwbfile.imaging_planes:
                imaging_plane = nwbfile.imaging_planes["ImagingPlane"]
                if hasattr(imaging_plane, "grid_spacing"):
                    grid_spacing = imaging_plane.grid_spacing[:]
                    pixel_size_um = grid_spacing[0]
                    field_metadata["pixel_size_um"] = pixel_size_um
                    # Add field dimensions if available and encoded as absolute sizes
                    if len(grid_spacing) >= 3 and pixel_size_um:
                        field_metadata["nxpix"] = int(round(grid_spacing[1] / (pixel_size_um * 1e-6)))
                        field_metadata["nypix"] = int(round(grid_spacing[2] / (pixel_size_um * 1e-6)))

            # Read spatial location data
            location_data = {}
            if "spatial" in nwbfile.processing:
                spatial_module = nwbfile.processing["spatial"]
                if "spatial_locations" in spatial_module.data_interfaces:
                    location_table = spatial_module["spatial_locations"]

                    for i in range(len(location_table)):
                        roi_id = location_table["roi_id"].data[i]
                        location_data[roi_id] = {
                            "ventral_dorsal_pos_um": location_table["ventral_dorsal_pos_um"].data[i],
                            "temporal_nasal_pos_um": location_table["temporal_nasal_pos_um"].data[i],
                            "field_ventral_dorsal_pos_um": location_table["field_ventral_dorsal_pos_um"].data[i],
                            "field_temporal_nasal_pos_um": location_table["field_temporal_nasal_pos_um"].data[i],
                        }

            # Read condition-specific data - THIS IS THE KEY FIX
            condition_modules = [name for name in nwbfile.processing.keys() if name.startswith("Condition_")]
            logger.info(f"Found condition modules: {condition_modules}")

            # Build the comprehensive table with one row per ROI per condition
            rows = []

            for condition_module_name in condition_modules:
                condition = condition_module_name.replace("Condition_", "")
                logger.info(f"Processing condition: {condition}")
                condition_module = nwbfile.processing[condition_module_name]

                # Read cell type data for this condition
                celltype_data = {}
                if "cell_types" in condition_module.data_interfaces:
                    celltype_table = condition_module["cell_types"]
                    for i in range(len(celltype_table)):
                        roi_id = celltype_table["roi_id"].data[i]

                        # Build celltype data dict with all available fields
                        roi_celltype_data = {}

                        # Add Baden16 trace preprocessing fields if available
                        for field_name in ["preproc_chirp", "preproc_bar"]:
                            if field_name in celltype_table.colnames:
                                roi_celltype_data[field_name] = celltype_table[field_name].data[i]

                        # Add new cell type classification fields
                        type_fields = [
                            "cluster_id",
                            "group_id",
                            "supergroup",
                            "prob_cluster",
                            "prob_group",
                            "prob_supergroup",
                            "prob_class",
                            "probs_per_cluster",
                        ]
                        for type_field in type_fields:
                            if type_field in celltype_table.colnames:
                                type_data = celltype_table[type_field].data[i]
                                if type_field in ["cluster_id", "group_id"]:
                                    type_data = int(type_data)
                                roi_celltype_data[type_field] = type_data

                        celltype_data[roi_id] = roi_celltype_data

                # Read receptive field data for this condition
                rf_data = {}
                if "receptive_fields" in condition_module.data_interfaces:
                    rf_table = condition_module["receptive_fields"]
                    for i in range(len(rf_table)):
                        roi_id = rf_table["roi_id"].data[i]
                        rf_data[roi_id] = {
                            "rf_cdia_um": rf_table["rf_cdia_um"].data[i],
                            "center_index": rf_table["center_index"].data[i],
                            "surround_index": rf_table["surround_index"].data[i],
                            "rf_gauss_qidx": rf_table["rf_gauss_qidx"].data[i],
                            "tRF_lag": rf_table["tRF_lag"].data[i],
                            "tRF_width": rf_table["tRF_width"].data[i],
                            "tRF_tRI": rf_table["tRF_tRI"].data[i],
                        }
                        # Optional advanced RF fields if present
                        if "split_qidx" in rf_table.colnames:
                            rf_data[roi_id]["split_qidx"] = rf_table["split_qidx"].data[i]
                        if "srf" in rf_table.colnames:
                            rf_data[roi_id]["srf"] = rf_table["srf"].data[i]
                        if "trf" in rf_table.colnames:
                            rf_data[roi_id]["trf"] = rf_table["trf"].data[i]
                        if "polarity" in rf_table.colnames:
                            rf_data[roi_id]["polarity"] = rf_table["polarity"].data[i]
                        if "trf_peak_idxs" in rf_table.colnames:
                            rf_data[roi_id]["trf_peak_idxs"] = strip_nan_trailing(rf_table["trf_peak_idxs"].data)[
                                i
                            ].astype(int)
                        if "relx_rf_roi_um" in rf_table.colnames:
                            rf_data[roi_id]["relx_rf_roi_um"] = rf_table["relx_rf_roi_um"].data[i]
                        if "rely_rf_roi_um" in rf_table.colnames:
                            rf_data[roi_id]["rely_rf_roi_um"] = rf_table["rely_rf_roi_um"].data[i]

                # Read stimulus features for this condition
                stimulus_features = {}
                for stimulus in ["chirp", "movingbar"]:
                    if f"{stimulus}_features" in condition_module.data_interfaces:
                        feature_table = condition_module[f"{stimulus}_features"]

                        # Process each ROI in the feature table
                        for i in range(len(feature_table)):
                            roi_id = feature_table["roi_id"].data[i]

                            if roi_id not in stimulus_features:
                                stimulus_features[roi_id] = {}

                            # Extract all columns except roi_id and roi_table_link
                            for col_name in feature_table.colnames:
                                if col_name not in ["roi_id", "roi_table_link", "id"]:
                                    # Fix column naming to match original table
                                    if stimulus == "chirp":
                                        # Remove duplicate "chirp_" prefix
                                        clean_name = col_name.replace("chirp_", "")
                                        if clean_name in ["on_off_index", "transience_index", "qidx", "pres_qidx"]:
                                            stimulus_features[roi_id][f"chirp_{clean_name}"] = feature_table[
                                                col_name
                                            ].data[i]
                                    elif stimulus == "movingbar":
                                        # Remove duplicate "bar_" prefix and map to original names
                                        clean_name = col_name.replace("bar_", "")
                                        if clean_name in [
                                            "ds_index",
                                            "ds_pvalue",
                                            "pref_dir",
                                            "os_index",
                                            "os_pvalue",
                                            "pref_or",
                                            "qidx",
                                            "pres_qidx",
                                        ]:
                                            stimulus_features[roi_id][f"bar_{clean_name}"] = feature_table[
                                                col_name
                                            ].data[i]

                # Extract time series data for this condition
                time_series_data = {}

                # Extract chirp averages
                if "chirp_60Hz_average" in condition_module.data_interfaces:
                    avg_interface = condition_module["chirp_60Hz_average"]
                    avg_data = avg_interface.data[:]
                    avg_timestamps = avg_interface.timestamps[:]
                    if len(avg_timestamps) > 1:
                        dt_value = avg_timestamps[1] - avg_timestamps[0]
                    else:
                        dt_value = np.nan
                    for roi_id in roi_ids_sorted:
                        col_idx = roi_id_to_ts_index.get(roi_id)
                        if col_idx is None or col_idx >= avg_data.shape[1]:
                            continue
                        if roi_id not in time_series_data:
                            time_series_data[roi_id] = {}
                        time_series_data[roi_id]["chirp_60Hz_average_norm"] = avg_data[:, col_idx]
                        time_series_data[roi_id]["chirp_60Hz_average_dt"] = dt_value

                if "chirp_8Hz_average" in condition_module.data_interfaces:
                    avg_interface = condition_module["chirp_8Hz_average"]
                    avg_data = avg_interface.data[:]
                    avg_timestamps = avg_interface.timestamps[:]
                    if len(avg_timestamps) > 1:
                        dt_value = avg_timestamps[1] - avg_timestamps[0]
                    else:
                        dt_value = np.nan
                    for roi_id in roi_ids_sorted:
                        col_idx = roi_id_to_ts_index.get(roi_id)
                        if col_idx is None or col_idx >= avg_data.shape[1]:
                            continue
                        if roi_id not in time_series_data:
                            time_series_data[roi_id] = {}
                        time_series_data[roi_id]["chirp_8Hz_average_norm"] = avg_data[:, col_idx]
                        time_series_data[roi_id]["chirp_8Hz_average_dt"] = dt_value

                # Extract chirp snippets and snippets_t0
                if "chirp_snippets" in condition_module.data_interfaces:
                    snippet_interface = condition_module["chirp_snippets"]
                    snippet_data = snippet_interface.data[:]
                    snippet_timestamps = snippet_interface.timestamps[:]
                    if len(snippet_timestamps) > 1:
                        dt_value = np.round(snippet_timestamps[1] - snippet_timestamps[0], 3)
                    else:
                        dt_value = np.nan
                    for roi_id in roi_ids_sorted:
                        col_idx = roi_id_to_ts_index.get(roi_id)
                        if col_idx is None or col_idx >= snippet_data.shape[1]:
                            continue
                        if roi_id not in time_series_data:
                            time_series_data[roi_id] = {}
                        time_series_data[roi_id]["chirp_snippets"] = snippet_data[:, col_idx]
                        time_series_data[roi_id]["chirp_snippets_dt"] = dt_value

                # Extract chirp snippets_t0 from separate table (new format)
                if "chirp_snippets_t0" in condition_module.data_interfaces:
                    snippets_t0_table = condition_module["chirp_snippets_t0"]
                    if hasattr(snippets_t0_table, "colnames") and "snippets_t0_array" in snippets_t0_table.colnames:
                        snippets_t0_data = snippets_t0_table["snippets_t0_array"].data[0]  # First (and only) row
                        # snippets_t0_data is (n_presentations, n_rois) - transpose to (n_rois, n_presentations)
                        snippets_t0_transposed = snippets_t0_data.T  # Now (n_rois, n_presentations)
                        for roi_id in roi_ids_sorted:
                            col_idx = roi_id_to_ts_index.get(roi_id)
                            if col_idx is None or col_idx >= snippets_t0_transposed.shape[0]:
                                continue
                            if roi_id not in time_series_data:
                                time_series_data[roi_id] = {}
                            time_series_data[roi_id]["chirp_snippets_t0"] = snippets_t0_transposed[col_idx]

                elif "chirp_snippets" in condition_module.data_interfaces:
                    # Fallback: extract from snippets timestamps
                    snippet_interface = condition_module["chirp_snippets"]
                    snippet_timestamps = snippet_interface.timestamps[:]
                    for roi_id in roi_ids_sorted:
                        if roi_id not in time_series_data:
                            time_series_data[roi_id] = {}
                        # Take the value only for the first scan entry
                        time_series_data[roi_id]["chirp_snippets_t0"] = snippet_timestamps[:, 0]

                # Extract chirp trigger times from TimeSeries metadata
                if "chirp_triggertimes_snippets" in condition_module.data_interfaces:
                    trigger_interface = condition_module["chirp_triggertimes_snippets"]

                    # Check if it's a TimeSeries (new format) or DynamicTable (old format)
                    if hasattr(trigger_interface, "data"):
                        # New format: TimeSeries with 3D data (n_triggers, n_rois, n_presentations)
                        trigger_data_3d = trigger_interface.data[:]

                        # Extract trigger times for each ROI using proper indexing
                        for roi_id in roi_ids_sorted:
                            col_idx = roi_id_to_ts_index.get(roi_id)
                            if col_idx is None or col_idx >= trigger_data_3d.shape[1]:
                                continue
                            if roi_id not in time_series_data:
                                time_series_data[roi_id] = {}
                            # Get trigger times for this ROI: shape (n_triggers, n_presentations)
                            roi_trigger_times = trigger_data_3d[:, col_idx, :]
                            time_series_data[roi_id]["chirp_triggertimes_snippets"] = roi_trigger_times

                # Extract bar snippets and snippets_t0
                if "movingbar_snippets" in condition_module.data_interfaces:
                    snippet_interface = condition_module["movingbar_snippets"]
                    snippet_data = snippet_interface.data[:]
                    snippet_timestamps = snippet_interface.timestamps[:]
                    if len(snippet_timestamps) > 1:
                        dt_value = np.round(snippet_timestamps[1] - snippet_timestamps[0], 3)
                    else:
                        dt_value = np.nan
                    for roi_id in roi_ids_sorted:
                        col_idx = roi_id_to_ts_index.get(roi_id)
                        if col_idx is None or col_idx >= snippet_data.shape[1]:
                            continue
                        if roi_id not in time_series_data:
                            time_series_data[roi_id] = {}
                        time_series_data[roi_id]["bar_snippets"] = snippet_data[:, col_idx]
                        time_series_data[roi_id]["bar_snippets_dt"] = dt_value

                # Extract bar snippets_t0 from separate table (new format)
                if "movingbar_snippets_t0" in condition_module.data_interfaces:
                    snippets_t0_table = condition_module["movingbar_snippets_t0"]
                    if hasattr(snippets_t0_table, "colnames") and "snippets_t0_array" in snippets_t0_table.colnames:
                        snippets_t0_data = snippets_t0_table["snippets_t0_array"].data[0]  # First (and only) row
                        # snippets_t0_data is (n_presentations, n_rois) - transpose to (n_rois, n_presentations)
                        snippets_t0_transposed = snippets_t0_data.T  # Now (n_rois, n_presentations)
                        for roi_id in roi_ids_sorted:
                            col_idx = roi_id_to_ts_index.get(roi_id)
                            if col_idx is None or col_idx >= snippets_t0_transposed.shape[0]:
                                continue
                            if roi_id not in time_series_data:
                                time_series_data[roi_id] = {}
                            time_series_data[roi_id]["bar_snippets_t0"] = snippets_t0_transposed[col_idx]
                elif "movingbar_snippets" in condition_module.data_interfaces:
                    # Fallback: extract from snippets timestamps
                    snippet_interface = condition_module["movingbar_snippets"]
                    snippet_timestamps = snippet_interface.timestamps[:]
                    for roi_id in roi_ids_sorted:
                        if roi_id not in time_series_data:
                            time_series_data[roi_id] = {}
                        time_series_data[roi_id]["bar_snippets_t0"] = snippet_timestamps

                # Extract bar trigger times from TimeSeries metadata
                if "movingbar_triggertimes_snippets" in condition_module.data_interfaces:
                    trigger_interface = condition_module["movingbar_triggertimes_snippets"]

                    # Check if it's a TimeSeries (new format) or DynamicTable (old format)
                    if hasattr(trigger_interface, "data"):
                        # New format: TimeSeries with 3D data (n_triggers, n_rois, n_presentations)
                        trigger_data_3d = trigger_interface.data[:]

                        # Extract trigger times for each ROI using proper indexing
                        for roi_id in roi_ids_sorted:
                            col_idx = roi_id_to_ts_index.get(roi_id)
                            if col_idx is None or col_idx >= trigger_data_3d.shape[1]:
                                continue
                            if roi_id not in time_series_data:
                                time_series_data[roi_id] = {}
                            # Get trigger times for this ROI: shape (n_triggers, n_presentations)
                            roi_trigger_times = trigger_data_3d[:, col_idx, :]
                            time_series_data[roi_id]["bar_triggertimes_snippets"] = roi_trigger_times

                # Create one row per ROI for this condition
                for roi_id in roi_ids_sorted:
                    row = {
                        "experimenter": experimenter,
                        "date": date,
                        "exp_num": int(exp_num) if isinstance(exp_num, str) and exp_num.isdigit() else exp_num,
                        "field": field,
                        "region": region,  # Add region information from folder structure
                        "cond1": condition,  # Add condition information
                        "roi_id": roi_id,
                    }

                    # Add experiment metadata
                    row.update(experiment_metadata)

                    # Add field metadata
                    row.update(field_metadata)

                    # Add ROI data
                    if roi_id in roi_data:
                        row.update(roi_data[roi_id])

                    # Add cell type data for this condition
                    if roi_id in celltype_data:
                        row.update(celltype_data[roi_id])

                    # Add location data
                    if roi_id in location_data:
                        row.update(location_data[roi_id])

                    # Add receptive field data for this condition
                    if roi_id in rf_data:
                        row.update(rf_data[roi_id])

                    # Add stimulus features for this condition
                    if roi_id in stimulus_features:
                        row.update(stimulus_features[roi_id])

                    # Add time series data for this condition
                    if roi_id in time_series_data:
                        row.update(time_series_data[roi_id])

                    rows.append(row)

            if not rows:
                logger.warning(f"No data could be reconstructed from {nwb_file_path}")
                return None

            df = pd.DataFrame(rows)

            # Enforce types and certain columns' precision.
            if "cluster_id" in df.columns:
                df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").astype("Int64")

            if "group_id" in df.columns:
                df["group_id"] = pd.to_numeric(df["group_id"], errors="coerce").astype("Int64")

            # Add file path for reference
            df["nwb_file_path"] = nwb_file_path

            logger.info(f"Successfully reconstructed comprehensive table from {nwb_file_path}: {len(df)} rows")
            return df

    except Exception as e:
        logger.error(f"Error reading {nwb_file_path}: {e}")
        import traceback

        traceback.print_exc()
        return None


def filter_nwb_table_to_match_original(nwb_table: pd.DataFrame) -> pd.DataFrame:
    """Filter full NWB table to match original export.

    Practically, this function filters out rows where celltype assignment columns are missing.

    Args:
        nwb_table: DataFrame loaded from NWB export

    Returns:
        Filtered DataFrame matching the original export structure
    """
    # Celltype assignment columns that must be present
    celltype_cols = [
        "preproc_chirp",
        "preproc_bar",
        "cluster_id",
        "group_id",
        "supergroup",
        "prob_cluster",
        "prob_group",
        "prob_supergroup",
        "prob_class",
        "probs_per_cluster",
    ]

    # Filter to only columns that exist in the table
    celltype_cols_present = [c for c in celltype_cols if c in nwb_table.columns]

    if not celltype_cols_present:
        logger.warning(
            "None of the required celltype columns found in table. "
            "Skipping filtering. Please verify the table structure."
        )
        return nwb_table

    # Filter: keep only rows where all celltype columns are present (not NaN)
    filtered_table = nwb_table[nwb_table[celltype_cols_present].notna().all(axis=1)].copy().reset_index(drop=True)

    rows_removed = len(nwb_table) - len(filtered_table)
    if rows_removed > 0:
        logger.info(f"Filtered out {rows_removed} rows missing celltype assignment data")

    return filtered_table


def combine_nwb_tables_from_directory(
    base_directory: str,
    pattern: str = "**/*.nwb",
    max_files: Optional[int] = None,
    filter_to_match_original: bool = True,
) -> pd.DataFrame:
    """Combine comprehensive tables from all NWB files in a directory structure.

    Args:
        base_directory: Base directory containing NWB files
        pattern: Glob pattern to find NWB files (default: all .nwb files recursively)
        max_files: Maximum number of files to process (for testing)
        filter_to_match_original: If True, filter the table to match the original DataJoint export
            by excluding rows missing celltype assignment data (default: True)

    Returns:
        Combined DataFrame containing data from all NWB files
    """
    from joblib import Parallel, delayed

    logger.info(f"Searching for NWB files in {base_directory} with pattern {pattern}")

    # Find all NWB files
    nwb_files = glob.glob(os.path.join(base_directory, pattern), recursive=True)

    if max_files is not None:
        nwb_files = nwb_files[:max_files]
        logger.info(f"Limiting to first {max_files} files for testing")

    logger.info(f"Found {len(nwb_files)} NWB files")

    # Process files in parallel with max 8 processes and tqdm progress bar
    logger.info("Processing NWB files in parallel (max 8 processes)")

    # Create a tqdm progress bar
    with tqdm(total=len(nwb_files), desc="Reading NWB files") as pbar:

        def update_progress(result):
            pbar.update(1)
            return result

        all_dataframes = []

        # Process in batches to show progress
        batch_size = max(1, len(nwb_files) // 20)  # 20 progress updates
        for i in range(0, len(nwb_files), batch_size):
            batch_files = nwb_files[i : i + batch_size]
            batch_results = Parallel(n_jobs=8)(
                delayed(read_comprehensive_table_from_nwb)(nwb_file) for nwb_file in batch_files
            )
            all_dataframes.extend(batch_results)
            pbar.update(len(batch_files))

    # Filter out None results and count successful reads
    successful_dataframes = [df for df in all_dataframes if df is not None]
    successful_reads = len(successful_dataframes)
    failed_reads = len(nwb_files) - successful_reads

    if failed_reads > 0:
        logger.warning(f"Failed to read data from {failed_reads} NWB files")

    if not successful_dataframes:
        logger.error("No data could be read from any NWB files")
        return pd.DataFrame()

    # Combine all DataFrames
    logger.info(f"Combining data from {successful_reads} successful reads")
    combined_df = pd.concat(successful_dataframes, ignore_index=True)

    # Clean up the DataFrame similar to extraction_table.py
    combined_df = combined_df.infer_objects()

    # Drop columns that are not needed for analysis (similar to extraction_table.py)
    drop_columns = [
        "nwb_file_path",  # This was added for reference but not needed in final table
        "field_name",  # Redundant with field column
    ]
    drop_columns = [col for col in drop_columns if col in combined_df.columns]

    if drop_columns:
        combined_df = combined_df.drop(columns=drop_columns)

    # Optionally filter to match original export
    if filter_to_match_original:
        logger.info("Filtering table to match original DataJoint export...")
        combined_df = filter_nwb_table_to_match_original(combined_df)

    logger.info(f"Combined table shape: {combined_df.shape}")
    return combined_df


def save_reconstructed_table(nwb_table: pd.DataFrame, output_path: str, format: str = "parquet") -> None:
    """Save the reconstructed table to a file.

    Args:
        nwb_table: DataFrame to save
        output_path: Path where to save the file
        format: File format ("pickle", "csv", "parquet")
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if format == "pickle":
        nwb_table.to_pickle(output_path)
    elif format == "csv":
        nwb_table.to_csv(output_path, index=False)
    elif format == "parquet":
        nwb_table_serialized = serialize_numpy_arrays(nwb_table)
        nwb_table_serialized.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Saved reconstructed table to {output_path}")


def main(
    nwb_base_directory: str,
    output_directory: str = "./reconstructed_tables",
    max_files: Optional[int] = None,
    filter_to_match_original: bool = False,
) -> None:
    """Main function to reconstruct tables from NWB files.

    Args:
        nwb_base_directory: Directory containing exported NWB files
        output_directory: Directory to save reconstructed tables
        max_files: Maximum number of NWB files to process (for testing)
        filter_to_match_original: If True, filter table to match original DataJoint export
    """
    logging.basicConfig(level=logging.INFO)

    # Reconstruct the comprehensive table
    logger.info("Starting table reconstruction from NWB files")
    combined_table = combine_nwb_tables_from_directory(
        nwb_base_directory, max_files=max_files, filter_to_match_original=filter_to_match_original
    )

    if combined_table.empty:
        logger.error("No data was reconstructed")
        return

    # Save reconstructed table
    os.makedirs(output_directory, exist_ok=True)

    output_pickle = os.path.join(output_directory, "reconstructed_all_RGC_table.pkl")
    save_reconstructed_table(combined_table, output_pickle, format="pickle")

    output_parquet = os.path.join(output_directory, "reconstructed_all_RGC_table.parquet")
    save_reconstructed_table(combined_table, output_parquet, format="parquet")

    logger.info(f"Table reconstruction completed. Output saved to {output_directory}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reconstruct comprehensive analysis table from NWB files")
    parser.add_argument("nwb_directory", help="Directory containing NWB files")
    parser.add_argument(
        "--output", "-o", default="./reconstructed_tables", help="Output directory for reconstructed tables"
    )
    parser.add_argument("--max-files", "-m", type=int, help="Maximum number of files to process (for testing)")
    parser.add_argument(
        "--filter-to-match-original",
        "-f",
        action="store_true",
        help="Filter table to match original DataJoint export (exclude rows missing celltype assignment data)",
    )

    args = parser.parse_args()

    main(
        nwb_base_directory=args.nwb_directory,
        output_directory=args.output,
        max_files=args.max_files,
        filter_to_match_original=args.filter_to_match_original,
    )
