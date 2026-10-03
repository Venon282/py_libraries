"""Unit tests for batchIterable."""

import itertools

import pytest

from py_libraries.type.iterable import batchIterable


@pytest.mark.parametrize(
    ('items', 'batch_size', 'expected'),
    [
        ([1, 2, 3, 4, 5], 2, [[1, 2], [3, 4], [5]]),
        ([1, 2, 3, 4], 2, [[1, 2], [3, 4]]),
        ([1, 2], 5, [[1, 2]]),
        ([1, 2, 3], 1, [[1], [2], [3]]),
        ([1, 2, 3], 3, [[1, 2, 3]]),
        ([], 3, []),
    ],
)
def test_batches(items: list[int], batch_size: int, expected: list[list[int]]) -> None:
    """Split the items into consecutive batches, the last one possibly shorter.

    Args:
        items: Source items.
        batch_size: Number of items per batch.
        expected: Expected batches.
    """
    assert list(batchIterable(items, batch_size)) == expected


@pytest.mark.parametrize('n_items', range(0, 12))
@pytest.mark.parametrize('batch_size', [1, 2, 3, 5, 20])
def test_batches_concatenate_back_to_the_input(n_items: int, batch_size: int) -> None:
    """Rebuild the input by chaining the batches, all full except the last one.

    Args:
        n_items: Number of source items.
        batch_size: Number of items per batch.
    """
    items = list(range(n_items))

    batches = list(batchIterable(items, batch_size))

    assert list(itertools.chain.from_iterable(batches)) == items
    assert all(len(batch) == batch_size for batch in batches[:-1])
    assert all(isinstance(batch, list) for batch in batches)
    if batches:
        assert 1 <= len(batches[-1]) <= batch_size


def test_batch_size_can_be_passed_by_keyword() -> None:
    """Accept batch_size as a keyword argument."""
    assert list(batchIterable([1, 2, 3], batch_size=2)) == [[1, 2], [3]]


def test_accepts_a_one_shot_iterator() -> None:
    """Batch an iterator that can only be consumed once."""
    assert list(batchIterable(iter('abcde'), 2)) == [['a', 'b'], ['c', 'd'], ['e']]


def test_consumes_the_source_lazily() -> None:
    """Read only one batch worth of items per iteration step."""
    endless = itertools.count()

    batches = batchIterable(endless, 3)

    assert next(batches) == [0, 1, 2]
    assert next(batches) == [3, 4, 5]


@pytest.mark.parametrize('batch_size', [0, -1, -5])
def test_batch_size_below_one_raises_value_error_on_first_iteration(
    batch_size: int,
) -> None:
    """Raise ValueError when the first batch is requested, not at creation.

    Args:
        batch_size: Invalid number of items per batch.
    """
    batches = batchIterable([1, 2, 3], batch_size)

    with pytest.raises(ValueError, match='at least 1'):
        next(batches)


def test_batch_size_below_one_raises_even_for_an_empty_iterable() -> None:
    """Reject an invalid batch_size before looking at the source items."""
    with pytest.raises(ValueError, match='at least 1'):
        list(batchIterable([], 0))
