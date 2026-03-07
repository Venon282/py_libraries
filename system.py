import os
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from itertools import chain
import concurrent.futures
import platform
import sys
from functools import cache

is_windows = platform.system().lower().startswith("win")
if is_windows:
    import msvcrt
else:
    import select

#? use shutil.rmtree
# def deleteAll(path:str):
#     for root, dirs, files in os.walk(path, topdown=False):
#         for file in files:
#             os.remove(os.path.join(root, file))  
#         for dir in dirs:
#             os.rmdir(os.path.join(root, dir))
            
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

def _filesToNpy(iterator, replace: bool = False):
    for file in iterator:
        try:
            npy_file = file.with_suffix('.npy')
            data = np.loadtxt(file)
            np.save(npy_file, data)
            if replace:
                file.unlink()
        except Exception as e:
            logging.error(f'with the file: {file.as_posix()}\n{e}')
                
def filesToNpy(
    source: str | Path | list[str | Path],
    recursive: bool = False,
    replace: bool = False,
    extensions: str | list | tuple = '.txt',
    progress_bar: bool = False,
    max_workers: int|None = 0,
    chunk_size: int = 1
):
    """
    Convert files into NumPy .npy files.

    Parameters
    ----------
    source : str | Path | list[str | Path]
        Either:
          - A directory path (all matching files inside will be converted).
          - A single file path.
          - A list of file paths.
    recursive : bool, default=False
        If True and source is a directory, search recursively for files.
    replace : bool, default=False
        If True, delete the original text file after conversion.
    extensions : str | list | tuple, default=".txt"
        File extension(s) to look for (e.g. ".txt" or [".txt", ".csv"]).
    progress_bar : bool, default=False
        If True, show a progress bar using tqdm.

    Notes
    -----
    - Files are loaded with `numpy.loadtxt`, so they must contain numeric data.
    - If you have plain-text files (non-numeric), modify the loading logic.
    """
    files = []
    
    # Normalize extensions to tuple
    if isinstance(extensions, str):
        extensions = (extensions,)
    elif isinstance(extensions, list):
        extensions = tuple(extensions)
    
    # Source is a folder
    if isinstance(source, (str, Path)):
        folder = Path(source)
        source = str(source)
        if folder.is_dir():
            search_function = folder.rglob if recursive else folder.glob
            files = list(chain.from_iterable(search_function(f"*{extension}") for extension in extensions))
        elif folder.is_file() and source.endswith(extensions):
            files = [folder]
        else:
            raise ValueError(f"Provided a path that is not a directory or file with the good extention: {folder.as_posix()}")
    # Source is a list
    elif isinstance(source, list):
        files = list(map(Path, source)) 
    else: 
        raise TypeError("source must be a folder path or a list of file paths")
    
    if max_workers == 0:
        iterator = files if not progress_bar else tqdm(files, mininterval=1, desc='Converting files', unit='file')
        _filesToNpy(iterator=iterator, replace=replace)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
            futures = [executor.submit(_filesToNpy, chunk, replace) for chunk in chunks]
            iterator = concurrent.futures.as_completed(futures) 
            if progress_bar: 
                iterator = tqdm(iterator, total=len(futures), mininterval=1, desc='Converting files', unit='chunk')
            for future in iterator:
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error during file conversion: {e}")
            
            
def filesDifferents(dir1: str|Path, dir2: str|Path, same_extension: bool=True):
    def getFiles(directory: str|Path):
        with os.scandir(directory) as it:
            if same_extension:
                return {entry.name for entry in it if entry.is_file()}
            else:
                return {Path(entry.name).stem for entry in it if entry.is_file()}
        
    dir1_files = getFiles(dir1)
    dir2_files = getFiles(dir2)

    return dir1_files - dir2_files
    

def inputNoBlocking(buffer=['']):
    """
    Non-blocking console input that returns a full line once Enter is pressed.
    Otherwise returns None.
    Keeps internal buffer between calls (via default arg trick).
    """

    # Windows version
    if is_windows:
        while msvcrt.kbhit():
            c = msvcrt.getwch()
            if c in ('\r', '\n'):  # Enter
                line = buffer[0]
                buffer[0] = ''
                print(line, end=' ' * 20)
                return line.strip()
            elif c == '\b':  # Backspace
                buffer[0] = buffer[0][:-1]
                print(buffer[0], end=' ' * 20 + '\r')
            else:
                buffer[0] += c
                print(buffer[0], end=' ' * 20 + '\r')
        return None

    # Unix / Linux / macOS version
    else:
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip()
            return line
        return None
    
def ipAddressesFromStrNumber(s: str) -> list[str]:
    """ 
    Given a string s containing only digits, return all possible valid IP addresses that can be formed by inserting dots into s. 

    Example 1:
    Input: s = "25525511135"
    Output: ["255.255.11.135","255.255.111.35"]
    
    Example 2:
    Input: s = "0000"
    Output: ["0.0.0.0"]
    
    Example 3:
    Input: s = "101023"
    Output: ["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]
"""
    @cache
    def isNotValid(s):
        return (int(s) > 255) or (s[0] == '0' and (s.count('0') != len(s) or len(s)!=1))

    if len(s) < 4 or len(s) > 12:
        return []

    adresses = []

    for p1 in range(1, len(s)-2):
        num1 =  s[:p1]
        if isNotValid(num1):
            return adresses

        for p2 in range(p1+1, len(s)-1):
            num2 = s[p1:p2]
            if isNotValid(num2):
                break

            for p3 in range(p2+1, len(s)):
                num3 = s[p2:p3]
                if isNotValid(num3):
                    break

                num4 = s[p3:]
                if isNotValid(num4):
                    continue

                adresses.append(f'{num1}.{num2}.{num3}.{num4}')
    return adresses