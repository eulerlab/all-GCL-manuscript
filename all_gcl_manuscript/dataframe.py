import numpy as np
import pandas as pd


def filter_df(
        df: pd.DataFrame,
        condition_filter=True,
        quality_filter=False,
        location_filter=False,
        genline_filter=True,
        rf_quality_filer=False,
        verbose=True,
):
    n_tot = df.shape[0]
    if condition_filter:
        keep = df['cond1'].isin(['control', 'c1', 'C1'])
        if verbose:
            print(f'Filtering condition. Removing {sum(~keep)/n_tot:.0%} of the data.')
        df = df[keep]

    if quality_filter:
        keep = ((df.chirp_qidx > 0.35) | (df.bar_qidx > 0.6)) & (df.bar_pres_qidx > 1. / 3)
        if verbose:
            print(f'Filtering quality. Removing {sum(~keep)/n_tot:.0%} of the data.')
        df = df[keep]

    if location_filter:
        keep = (df.field_temporal_nasal_pos_um.notnull() & df.field_ventral_dorsal_pos_um.notnull())
        if verbose:
            print(f'Filtering location. Removing {sum(~keep)/n_tot:.0%} of the data.')
        df = df[keep]

    if genline_filter:
        keep = df.genline == 'Bl6'
        if verbose:
            print(f'Filtering genline. Removing {sum(~keep)/n_tot:.0%} of the data.')
        df = df[keep]

    if rf_quality_filer:
        keep = df.rf_cdia_um.notnull() & (df.rf_cdia_um < 600) & (df.rf_gauss_qidx > 0.4)
        if verbose:
            print(f'Filtering RF. Removing {sum(~keep)/n_tot:.0%} of the data.')
        df = df[keep]

    return df
