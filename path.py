import os
from itertools import islice

def _batched(iterable, batch_size=10):
    """Yield lists (batches) of size up to batch_size."""
    iterable = iter(iterable)
    
    while True:
        batch = list(islice(iterable, batch_size))
        if not batch:
            break
        yield batch
            
def filesPathRec(path):
    """Generator that yields file paths recursively."""
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file():
                yield entry.path
            elif entry.is_dir():
                yield from filesPathRec(entry.path)

def batchedFilesRec(path, batch_size=10):
     """Yield files in batches from a directory recursively."""
     return _batched(filesPathRec(path), batch_size=batch_size)