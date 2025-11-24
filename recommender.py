import joblib
import pandas as pd
from typing import List, Tuple


DEFAULT_MODEL_PATH = 'ctr_model.joblib'
DEFAULT_HASHER_PATH = 'ctr_hasher.joblib'

# The categorical columns used when the model was trained (kept in sync with previous scripts)
CATEGORICAL_COLS = [
    'hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
    'site_category', 'app_id', 'app_domain', 'app_category',
    'device_id', 'device_ip', 'device_model', 'device_type'
]


def load_model_and_hasher(model_path: str = DEFAULT_MODEL_PATH, hasher_path: str = DEFAULT_HASHER_PATH):
    """Load the saved model and feature hasher from disk.

    Returns (model, hasher). Raises FileNotFoundError with a helpful message if missing.
    """
    try:
        model = joblib.load(model_path)
        hasher = joblib.load(hasher_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Model or hasher file not found. Expected '{model_path}' and '{hasher_path}'.\n"
            "Please run training pipeline to produce these files or confirm paths.") from e

    return model, hasher


def _get_cols_for_hashing(df: pd.DataFrame, required_cols: List[str]) -> List[str]:
    """Return the intersection of required_cols and columns present in df.

    This makes the recommender tolerant to candidate files that carry fewer features.
    """
    present = [c for c in required_cols if c in df.columns]
    if not present:
        raise ValueError(f"None of the expected categorical columns found in candidate data. Expected one of: {required_cols}")
    return present


def score_candidates(df: pd.DataFrame, hasher, model, categorical_cols: List[str] = None) -> pd.DataFrame:
    """Score each row in df with the loaded model and return a DataFrame with a new 'click_prob' column.

    - df: candidates (each row is an ad impression to score)
    - hasher: the fitted hasher used to transform categorical dicts
    - model: the trained classifier with predict_proba
    - categorical_cols: optional list of columns to use (defaults to CATEGORICAL_COLS intersection)
    """
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS

    cols_to_use = _get_cols_for_hashing(df, categorical_cols)

    X_df = df[cols_to_use].astype(str)
    X_dict = X_df.to_dict('records')
    X_hashed = hasher.transform(X_dict)

    # predict_proba must be available
    probs = model.predict_proba(X_hashed)[:, 1]

    out = df.copy()
    out['click_prob'] = probs
    return out


def recommend(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return top_n rows sorted by predicted click probability (highest first)."""
    if 'click_prob' not in df.columns:
        raise ValueError("DataFrame must contain 'click_prob' column. Call score_candidates first.")
    return df.sort_values('click_prob', ascending=False).head(top_n)


def example_usage():
    """Small example that shows how to load and score a few candidates from test.csv.

    This function is intentionally lightweight so it can be used both by scripts and interactive runs.
    """
    import os

    model_path = DEFAULT_MODEL_PATH
    hasher_path = DEFAULT_HASHER_PATH
    candidates_path = 'test.csv'

    if not os.path.exists(candidates_path):
        print(f"Candidates file '{candidates_path}' not found in working directory.")
        return

    model, hasher = load_model_and_hasher(model_path, hasher_path)
    df = pd.read_csv(candidates_path, nrows=1000)
    scored = score_candidates(df, hasher, model)
    tops = recommend(scored, top_n=10)
    print(tops[['click_prob'] + [c for c in df.columns if c in CATEGORICAL_COLS]])


if __name__ == '__main__':
    example_usage()
