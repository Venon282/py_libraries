"""Helpers for iterables."""

import itertools
from collections.abc import Generator, Iterable
from typing import TypeVar

T = TypeVar('T')


def batchIterable(
    iterable: Iterable[T], batch_size: int
) -> Generator[list[T], None, None]:
    """Yield successive batches from an iterable.

    This function consumes the given iterable in chunks of size `batch_size`,
    returning each chunk as a list. It stops when the iterable is exhausted.

    Args:
        iterable: Any iterable source (e.g., list, generator, file lines).
        batch_size: Number of items per batch. Must be at least 1.

    Yields:
        A list containing up to `batch_size` items from the iterable.

    Raises:
        ValueError: If batch_size is lower than 1. Raised on the first iteration, not
            when the generator is created.

    Example:
        for group in batchIterable([1, 2, 3, 4, 5], 2):
            print(group)
        # Output: [1, 2], [3, 4], [5]
    """
    if batch_size < 1:
        raise ValueError(f'batch_size must be at least 1, got {batch_size}.')

    it = iter(iterable)

    while True:
        batch = list(itertools.islice(it, batch_size))

        if not batch:
            break

        yield batch
