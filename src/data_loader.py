"""
data_loader.py
---------------
Small, dependency-light helper for loading the security log CSV.

Kept separate from app.py so that later phases (preprocessing, anomaly
detection, etc.) can import the same loader instead of duplicating
pandas.read_csv() calls with slightly different logic scattered around.
"""

import os

import pandas as pd

from config import LOG_FILE_PATH


def load_logs(path: str = LOG_FILE_PATH) -> pd.DataFrame:
    """
    Load the synthetic security log CSV into a DataFrame.

    Parameters
    ----------
    path : str
        Path to the CSV file. Defaults to config.LOG_FILE_PATH.

    Returns
    -------
    pd.DataFrame
        The raw log data, with `timestamp` parsed as a datetime column.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist yet (e.g. generate_data.py hasn't been run).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Log file not found at '{path}'. Run `python generate_data.py` "
            "first to create the synthetic dataset."
        )

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def get_basic_stats(df: pd.DataFrame) -> dict:
    """
    Compute a few simple summary statistics used on the dashboard overview.
    Deliberately simple for Phase 0 — real anomaly/incident stats arrive
    in later phases.
    """
    return {
        "total_records": len(df),
        "unique_users": df["user"].nunique(),
        "unique_source_ips": df["source_ip"].nunique(),
        "date_range": (df["timestamp"].min(), df["timestamp"].max()),
        "event_type_counts": df["event_type"].value_counts().to_dict(),
    }
