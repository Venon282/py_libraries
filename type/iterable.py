import itertools

def batch(iterable, batch_size):
    """
    Yield successive batches from an iterable.

    This function consumes the given iterable in chunks of size `batch_size`,
    returning each chunk as a list. It stops when the iterable is exhausted.

    Args:
        iterable: Any iterable source (e.g., list, generator, file lines).
        batch_size (int): Number of items per batch.

    Yields:
        list: A list containing up to `batch_size` items from the iterable.

    Example:
        for group in batch([1, 2, 3, 4, 5], 2):
            print(group)
        # Output: [1, 2], [3, 4], [5]
    """
    it = iter(iterable)
    
    while True:
        batch = list(itertools.islice(it, batch_size))
        
        if not batch:
            break
        
        yield batch