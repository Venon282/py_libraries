import os
import ctypes
from contextlib import contextmanager

def deleteAll(path:str):
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))  
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
            
def displayFilesRec(path:str):
    for root, dirs, files in os.walk(path):
        print(root)
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
            
def unicDirsRec(path:str):
    lst = set()
    for _, dirs, _ in os.walk(path):
        for dir in dirs:
            lst.add(dir)
    return lst

def countFiles(directory:str):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

@contextmanager
def noSleep(display: bool = True):
    """
    Prevents Windows from going into standby mode while the `with` block is running.

    Args:
        display (bool): If True, also prevents the screen from turning off.
        
    Exemple:
        with noSleep(display=True):
            # my code
    """
    # Constant values
    ES_CONTINUOUS       = 0x80000000
    ES_SYSTEM_REQUIRED  = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002

    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    
    # Include the screen sleep ?
    if display:
        flags |= ES_DISPLAY_REQUIRED

    # activate "no sleep"
    ctypes.windll.kernel32.SetThreadExecutionState(flags)
    try:
        yield
    finally:
        # Restauration of the normal behavior
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        
    