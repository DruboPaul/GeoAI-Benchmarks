"""
Spatial Block Split Module — Drubo Live Benchmark Platform
===========================================================
Splits plot data into train/test sets using geographic grid blocks to prevent
spatial autocorrelation leakage.

Authors: Drubo / Antigravity
"""

import numpy as np
import pandas as pd


def spatial_block_split(df, n_rows=5, n_cols=4, test_fraction=0.3,
                        lat_col="lat", lon_col="long", random_state=42):
    """
    Assign each plot to a geographic grid cell, then allocate cells to
    train or test to achieve approximately the desired test fraction.

    Parameters
    ----------
    df : pd.DataFrame with lat/long columns
    n_rows, n_cols : grid dimensions
    test_fraction : target fraction of plots in test set
    lat_col, lon_col : column names for coordinates
    random_state : for reproducibility

    Returns
    -------
    df_train, df_test, split_info : DataFrames + metadata dict
    """
    rng = np.random.RandomState(random_state)

    lat_min, lat_max = df[lat_col].min(), df[lat_col].max()
    lon_min, lon_max = df[lon_col].min(), df[lon_col].max()

    # Add small buffer to include boundary points
    lat_range = lat_max - lat_min + 1e-6
    lon_range = lon_max - lon_min + 1e-6

    # Assign grid cell indices
    df = df.copy()
    df["_grid_row"] = np.clip(
        ((df[lat_col] - lat_min) / lat_range * n_rows).astype(int),
        0, n_rows - 1
    )
    df["_grid_col"] = np.clip(
        ((df[lon_col] - lon_min) / lon_range * n_cols).astype(int),
        0, n_cols - 1
    )
    df["_block_id"] = df["_grid_row"] * n_cols + df["_grid_col"]

    # Count plots per block
    block_counts = df["_block_id"].value_counts().to_dict()
    total_plots = len(df)
    target_test = int(total_plots * test_fraction)

    # Shuffle blocks and greedily assign to test until target reached
    block_ids = list(block_counts.keys())
    rng.shuffle(block_ids)

    test_blocks = set()
    test_count = 0
    for bid in block_ids:
        if test_count >= target_test:
            break
        test_blocks.add(bid)
        test_count += block_counts[bid]

    df["_split"] = df["_block_id"].apply(lambda b: "test" if b in test_blocks else "train")

    df_train = df[df["_split"] == "train"].drop(
        columns=["_grid_row", "_grid_col", "_block_id", "_split"]).reset_index(drop=True)
    df_test = df[df["_split"] == "test"].drop(
        columns=["_grid_row", "_grid_col", "_block_id", "_split"]).reset_index(drop=True)

    split_info = {
        "grid_shape": f"{n_rows}×{n_cols}",
        "total_blocks": n_rows * n_cols,
        "train_blocks": len(block_ids) - len(test_blocks),
        "test_blocks": len(test_blocks),
        "train_plots": len(df_train),
        "test_plots": len(df_test),
        "test_fraction_actual": round(len(df_test) / total_plots, 3),
        "lat_range": (round(lat_min, 4), round(lat_max, 4)),
        "lon_range": (round(lon_min, 4), round(lon_max, 4)),
        "test_block_ids": sorted(test_blocks),
        "block_counts": block_counts,
    }

    return df_train, df_test, split_info


def get_block_grid_for_viz(df, n_rows=5, n_cols=4, lat_col="lat", lon_col="long"):
    """
    Return grid cell boundaries for map visualization.
    Returns list of dicts with block_id, lat_min, lat_max, lon_min, lon_max, plot_count.
    """
    lat_min, lat_max = df[lat_col].min(), df[lat_col].max()
    lon_min, lon_max = df[lon_col].min(), df[lon_col].max()
    lat_range = lat_max - lat_min + 1e-6
    lon_range = lon_max - lon_min + 1e-6
    lat_step = lat_range / n_rows
    lon_step = lon_range / n_cols

    cells = []
    for r in range(n_rows):
        for c in range(n_cols):
            bid = r * n_cols + c
            cell_lat_min = lat_min + r * lat_step
            cell_lat_max = lat_min + (r + 1) * lat_step
            cell_lon_min = lon_min + c * lon_step
            cell_lon_max = lon_min + (c + 1) * lon_step

            count = len(df[
                (df[lat_col] >= cell_lat_min) & (df[lat_col] < cell_lat_max) &
                (df[lon_col] >= cell_lon_min) & (df[lon_col] < cell_lon_max)
            ])

            cells.append({
                "block_id": bid,
                "lat_min": round(cell_lat_min, 5),
                "lat_max": round(cell_lat_max, 5),
                "lon_min": round(cell_lon_min, 5),
                "lon_max": round(cell_lon_max, 5),
                "plot_count": count,
            })
    return cells
