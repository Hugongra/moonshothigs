from __future__ import annotations

import numpy as np
import pandas as pd

SENTINEL = -999.0
META_COLS = {"EventId", "Weight", "Label", "KaggleSet", "KaggleWeight"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def load_training_table(
    csv_path,
    *,
    max_rows: int | None = None,
    random_state: int = 42,
    shuffle_sample: bool = False,
) -> pd.DataFrame:
    """
    Load competition training subset (`KaggleSet == 't'`).

    Reads the full CSV then filters (matches typical challenge workflows). Use
    ``max_rows`` to keep only the first N **training** rows after filtering, or a
    random subsample when ``shuffle_sample=True``.
    """
    df = pd.read_csv(csv_path)
    df = df[df["KaggleSet"] == "t"].copy()
    if len(df) == 0:
        raise ValueError("No rows with KaggleSet=='t'. Check CSV.")

    if max_rows is not None and len(df) > max_rows:
        if shuffle_sample:
            df = df.sample(n=max_rows, random_state=random_state)
        else:
            df = df.iloc[:max_rows].copy()

    return df.reset_index(drop=True)


def X_y_weights(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cols = feature_columns(df)
    X = df[cols].replace(SENTINEL, np.nan).astype(np.float32)
    y = (df["Label"].values == "s").astype(np.int8)
    w = df["Weight"].astype(np.float64).values
    return X, y, w
