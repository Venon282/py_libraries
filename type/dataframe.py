from pathlib import Path
from typing import Any

import pandas as pd


def _getMetaPath(path: str | Path) -> Path:
    """Generate a metadata path for a given CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        Path to the corresponding metadata JSON file.
    """
    p = Path(path)
    return p.with_name(p.stem + "_meta.json")

def saveMultiIndexCsv(
    df: pd.DataFrame, path: str | Path, **kwargs: Any
) -> None:
    """Save a DataFrame with MultiIndex to CSV along with metadata.

    Args:
        df: DataFrame to save.
        path: Path to save the CSV file.
        **kwargs: Additional arguments to pass to pd.DataFrame.to_csv.
    """
    n_row_levels = len(df.index[0]) if isinstance(df.index, pd.MultiIndex) else 1
    n_col_levels = len(df.columns[0]) if isinstance(df.columns, pd.MultiIndex) else 1

    df.to_csv(path, **kwargs)

    # Save metadata
    pd.Series({"n_row_levels": n_row_levels, "n_col_levels": n_col_levels}).to_json(
        _getMetaPath(path)
    )
    
def loadMultiIndexCsv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a DataFrame with MultiIndex from CSV using stored metadata.

    Args:
        path: Path to the CSV file.
        **kwargs: Additional arguments to pass to pd.read_csv.

    Returns:
        DataFrame with MultiIndex rows and columns restored.
    """
    meta = pd.read_json(_getMetaPath(path), typ="series")
    return pd.read_csv(
        path,
        index_col=list(range(int(meta["n_row_levels"]))),
        header=list(range(int(meta["n_col_levels"]))),
        **kwargs,
    )