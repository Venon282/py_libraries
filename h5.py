import h5py
import numpy as np
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

def getDatasetsName(h5):
    return [name for name, obj in h5.items() if isinstance(obj, h5py.Dataset)]
    
def getDatasetsNameRec(h5):
    datasets = []
    
    def collect(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append(name)
            
    h5.visititems(collect)
    return datasets

def getDatasetsObject(h5):
    return [obj for name, obj in h5.items() if isinstance(obj, h5py.Dataset)]
    
def getDatasetsObjectRec(h5):
    datasets = []
    
    def collect(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append(obj)
            
    h5.visititems(collect)
    return datasets

def getDatasets(h5):
    return {name: obj for name, obj in h5.items() if isinstance(obj, h5py.Dataset)}
    
def getDatasetsRec(h5):
    datasets = {}
    
    def collect(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets[name] = obj
            
    h5.visititems(collect)
    return datasets

def getGroups(h5):
    return [name for name, obj in h5.items() if isinstance(obj, h5py.Group)]
    
def getGroupsRec(h5):
    groups = []
    
    def collect(name, obj):
        if isinstance(obj, h5py.Group):
            groups.append(name)
            
    h5.visititems(collect)
    return groups

def getGroupsWithDataset(h5):
    return [name for name, obj in h5.items() if isinstance(obj, h5py.Group) and haveDataset(obj)]

def getGroupsWithDatasetRec(h5):
    groups = []
    
    def collect(name, obj):
        if isinstance(obj, h5py.Group) and haveDataset(obj):
            groups.append(name)
    
    h5.visititems(collect)
    return groups

def getGroupsWithGroup(h5):
    return [name for name, obj in h5.items() if isinstance(obj, h5py.Group) and haveGroup(obj)]

def getGroupsWithGroupRec(h5):
    groups = []
    
    def collect(name, obj):
        if isinstance(obj, h5py.Group) and haveGroup(obj):
            groups.append(name)
    
    h5.visititems(collect)
    return groups

def getGroupsAndDatasetsNameRec(h5):
    groups = []
    datasets = []
    
    def collect(name, obj):
        if isinstance(obj, h5py.Group):
            groups.append(name)
        elif isinstance(obj, h5py.Dataset):
            datasets.append(name)
            
    h5.visititems(collect)
    return groups, datasets

def getGroupsAndDatasetsObjectRec(h5):
    groups = []
    datasets = []
    
    def collect(name, obj):
        if isinstance(obj, h5py.Group):
            groups.append(name)
        elif isinstance(obj, h5py.Dataset):
            datasets.append(obj)
            
    h5.visititems(collect)
    return groups, datasets

def getGroupsAndDatasetsRec(h5):
    groups = []
    datasets = {}
    
    def collect(name, obj):
        if isinstance(obj, h5py.Group):
            groups.append(name)
        elif isinstance(obj, h5py.Dataset):
            datasets[name] = obj
            
    h5.visititems(collect)
    return groups, datasets

def haveDataset(h5):
    for _, obj in h5.items():
        if isinstance(obj, h5py.Dataset):
            return True
    return False

def haveGroup(h5):
    for _, obj in h5.items():
        if isinstance(obj, h5py.Group):
            return True
    return False
    
def display(h5):
    h5.visit(print)
    

def displayWithDetails(h5):
    def displayInfo(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, shape={obj.shape}, dtype={obj.dtype}")
    
    h5.visititems(displayInfo)
    
def getDescribe(h5):
    from .lst import describeValues
    import pandas as pd
    df = {}
    
    def collect(name, obj):
        if isinstance(obj, h5py.Dataset):
            logger.debug(f'Processing {name}')
            df[name] = describeValues(obj[:])
            
    h5.visititems(collect)
    return pd.DataFrame.from_dict(df, orient='index')

def describe(h5):
    from .lst import describe
    
    def collect(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(name)
            describe(obj[:])
            print()
            
    h5.visititems(collect)

def iterate(
    h5: h5py.File | h5py.Group,
    path: str | list,
    sep: str = '/'
):
    """
    Recursively traverses an HDF5 file or group using a path string 
    with optional wildcards (* and **).
    
    Args:
        h5: An open h5py.File or h5py.Group object.
        path: A path string or list of
                  (e.g. 'raw/*', 'data/group1/**', 'raw/**/data', 'measurements').
        sep: Path separator (default: '/').
        
    Yields:
        Tuples of (internal_path, h5py object) where object is a Group or Dataset.
    """
    if isinstance(path, list):
        for sub_path in path:
            yield from iterate(h5=h5, path=sub_path, sep=sep)
        return
    
    # Split and clean path
    path_parts = [p for p in path.split(sep) if p]

    def _walk(node: h5py.File | h5py.Group, parts: list[str], root: str = ''):
        # Base case → no more parts : yield the current node
        if not parts:
            yield root.rstrip(sep), node
            return

        part = parts[0]

        # deep recursive elements case
        if part == '**':
            # browse recursively all the content
            for key, obj in node.items():
                if isinstance(obj, h5py.Group):
                    # Firstly, continue to go down with ** and stay at this parts level
                    yield from _walk(obj, parts, root + key + sep)
                    # Then continue the walk (delete **) if there are more parts after **
                    yield from _walk(obj, parts[1:], root + key + sep)
                else:
                    #If a dataset and parts == ** (we want it), yield it
                    if len(parts) == 1:
                        yield root + key, obj
            return

        # all elements case
        if part == '*':
            for key in node.keys():
                obj = node[key]
                yield from _walk(obj, parts[1:], root + key + sep)
            return
        
        # A list of elements
        if part.startswith('[') and part.endswith(']'):
            for key in part[1:-1].split(','):
                key = key.strip('\'\" ')
                if key in node:
                    yield from _walk(node[key], parts[1:], root + key + sep)
            return
        
        # A list of exclude elements 
        if part.startswith('~[') and part.endswith(']'):
            undesired_keys = [k.strip('\'\" ') for k in part[2:-1].split(',')]
            for key in node.keys():
                if key in undesired_keys:
                    continue 
                obj = node[key]
                yield from _walk(obj, parts[1:], root + key + sep)
            return

        # Normal case
        if part in node:
            obj = node[part]
            if len(parts) == 1:  # Last element → on yield
                yield root + part, obj
            elif isinstance(obj, h5py.Group):
                yield from _walk(obj, parts[1:], root + part + sep)

    yield from _walk(h5, path_parts, root='')
    
def dynamicSize(h5, name, values, dtype=None):
    # Ensure values are at least 1D
    if np.isscalar(values):
        values = np.array([values])
    else: values = np.array(values)
    
    values_shape = list(values.shape)
    
    if dtype is None and values.dtype.kind in {'U', 'S', 'O'}:
        dtype = h5py.string_dtype(encoding='utf-8')
        values = values.astype(str).tolist()
    
    if name not in h5:
        
        values_shape[0] = None  # allow unlimited rows
        
        h5.create_dataset(
            name,
            data=values,
            maxshape=tuple(values_shape),
            chunks=True,
            dtype=dtype
        )
    else:
        dataset = h5[name]
        dataset_size = dataset.shape[0]
        new_size = dataset_size + values_shape[0]
        dataset.resize(new_size, axis=0)
        if h5py.check_dtype(vlen=dataset.dtype) is str:
            if isinstance(values, np.ndarray):
                values = values.astype(str).tolist()
        dataset[dataset_size:new_size] = values
        
def toExcel(h5, excel_path='./h5_to_excel.xlsx', verbose=0, max_rows=1_048_576, max_cols=16_384):
    import pandas as pd
    
    with pd.ExcelWriter(excel_path) as writer:
        iterator = tqdm(iterate(h5, '**'), mininterval=1, desc='Dataset to excel sheet') if verbose else iterate(h5, '**')
        for path, obj in iterator:
            if isinstance(obj, h5py.Dataset):
                sheet_name = path.replace('/','_')
                for i in range(0, len(obj), max_rows):
                    for j in range(0, len(obj[0]) if obj.ndim > 1 else 1, max_cols):
                        pd.DataFrame(obj[:]).to_excel(writer, sheet_name=f'{sheet_name}_{i}_{j}', index=False)
                
def getName(h5, sep='/'):
    """Return the current group or dataset name
    """
    return h5.name.split(sep)[-1]

