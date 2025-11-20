import numpy as np
import pandas as pd


def strip_nan_trailing(array):
    """
    Strips trailing NaNs from each row of a 2D array.

    This is useful for retrieving original, variable-length arrays from a
    NaN-padded, stacked array.

    Args:
        arr (np.ndarray): A 2D NumPy array, where rows may have trailing NaNs.

    Returns:
        list of np.ndarray: A list of 1D arrays, with trailing NaNs removed
                            from each corresponding row of the input array.
    """
    if array.ndim != 2:
        raise ValueError("Input array must be 2-dimensional.")

    stripped_rows = []
    for row in array:
        # Find indices of all non-NaN values in the row
        non_nan_indices = np.where(~np.isnan(row))[0]

        # If there are any non-NaN values, slice the row up to the last one.
        # Otherwise, the row was all NaNs, so return an empty array.
        if non_nan_indices.size > 0:
            last_valid_index = non_nan_indices[-1]
            stripped_rows.append(row[: last_valid_index + 1])
        else:
            stripped_rows.append(np.array([]))

    return stripped_rows


def serialize_numpy_arrays(df):
    """Serialize multi-dimensional numpy arrays in dataframe columns for storage in parquet."""
    df_serialized = df.copy()

    for col in df_serialized.columns:
        if df_serialized[col].dtype == "object":
            # Check if this column contains numpy arrays
            sample_non_null = df_serialized[col].dropna()
            if len(sample_non_null) > 0:
                first_val = sample_non_null.iloc[0]
                if isinstance(first_val, np.ndarray):
                    print(f"Serializing numpy arrays in column: {col}")
                    # Convert numpy arrays to nested lists - needed to save N-dimensional arrays to parquet
                    df_serialized[col] = df_serialized[col].apply(
                        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
                    )

    return df_serialized


def restore_numpy_arrays(df):
    """Automatically detect and restore numpy arrays from nested structures."""
    df_restored = df.copy()

    def is_nested_array_structure(val):
        """Check if a value is a nested array structure that should be converted to numpy array."""

        # Check for numpy object arrays containing other arrays (from parquet loading)
        if isinstance(val, np.ndarray) and val.dtype == object and val.size > 0:
            flat_val = val.flatten()
            for item in flat_val:
                if isinstance(item, (np.ndarray, list)):
                    return True

        # Check for nested lists
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], (list, np.ndarray)):
            return True

        return False

    def convert_to_numpy_array(val):
        """Convert nested structure to proper numpy array."""

        # Handle numpy object arrays (from parquet)
        if isinstance(val, np.ndarray) and val.dtype == object:
            # Try to stack the arrays inside
            try:
                return np.stack(val)
            except Exception:
                # If stacking fails, convert to list first then to array
                try:
                    nested_list = [item.tolist() if isinstance(item, np.ndarray) else item for item in val]
                    return np.array(nested_list)
                except Exception:
                    return val

        # Handle nested lists
        if isinstance(val, list):
            try:
                return np.array(val)
            except Exception:
                return val

        if val is None or pd.isna(val):
            return val

        return val

    for col in df_restored.columns:
        if df_restored[col].dtype == "object":
            # Check if this column contains nested array structures
            sample_non_null = df_restored[col].dropna()
            if len(sample_non_null) > 0:
                first_val = sample_non_null.iloc[0]
                if is_nested_array_structure(first_val):
                    print(f"Restoring numpy arrays in column: {col}")
                    df_restored[col] = df_restored[col].apply(convert_to_numpy_array)

    return df_restored
