import itertools
from collections.abc import Iterable
from typing import Generator, TypeVar

T = TypeVar("T")

def batchIterable(
    iterable: Iterable[T], batchSize: int
) -> Generator[list[T], None, None]:
    """Yield successive batches from an iterable.

    This function consumes the given iterable in chunks of size `batchSize`,
    returning each chunk as a list. It stops when the iterable is exhausted.

    Args:
        iterable: Any iterable source (e.g., list, generator, file lines).
        batchSize: Number of items per batch.

    Yields:
        A list containing up to `batchSize` items from the iterable.

    Example:
        for group in batchIterable([1, 2, 3, 4, 5], 2):
            print(group)
        # Output: [1, 2], [3, 4], [5]
    """
    it = iter(iterable)

    while True:
        batch = list(itertools.islice(it, batchSize))

        if not batch:
            break

        yield batch