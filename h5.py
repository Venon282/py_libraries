import h5py
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
