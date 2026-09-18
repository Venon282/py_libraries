from pathlib import Path
import pandas as pd

def _metaPath(path) -> Path:
    p = Path(path)
    return p.with_name(p.stem + "_meta.json")

def saveMultiIndexCsv(df, path, **kwargs):
    n_row_levels = len(df.index[0]) if isinstance(df.index, pd.MultiIndex) else 1
    n_col_levels = len(df.columns[0]) if isinstance(df.columns, pd.MultiIndex) else 1
    
    df.to_csv(path, **kwargs)
    
    # Save metadata
    pd.Series({"n_row_levels": n_row_levels, "n_col_levels":  n_col_levels}) \
    .to_json(_metaPath(path))
    
def loadMultiIndexCsv(path, **kwargs):
    meta = pd.read_json(_metaPath(path), typ="series")
    return pd.read_csv(path,
                    index_col=list(range(int(meta["n_row_levels"]))),
                    header=list(range(int(meta["n_col_levels"]))),
                    **kwargs)