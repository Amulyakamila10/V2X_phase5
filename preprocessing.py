
import numpy as np
import pandas as pd

FEATURES = [f"PC_{i}" for i in range(1, 19)]


def prepare_features(data):
    """
    Prepare input for the Phase 5 Extra Trees model.

    The deployed model expects exactly:
    PC_1 ... PC_18
    """

    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Unsupported input format.")

    missing = [c for c in FEATURES if c not in df.columns]

    if missing:
        raise ValueError(
            "Input does not contain the required Phase 2 "
            f"features. Missing: {missing}"
        )

    X = df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if X.isna().any().any():
        raise ValueError(
            "Input contains missing/non-numeric PC feature values."
        )

    return X.to_numpy(dtype=np.float32)
