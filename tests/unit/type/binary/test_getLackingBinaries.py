"""Unit tests for getLackingBinaries."""

import numpy as np
import pytest

from py_libraries.type.binary import getLackingBinaries


@pytest.mark.parametrize(
    ('lst', 'length', 'expected'),
    [
        ([], 2, ['0b0', '0b1', '0b10', '0b11']),
        (['0b0', '0b1', '0b10', '0b11'], 2, []),
        (['0b1', '0b10'], 2, ['0b0', '0b11']),
        ([], 0, ['0b0']),
        (['0b0'], 0, []),
        ([], 1, ['0b0', '0b1']),
    ],
)
def test_explicit_length(lst: list[str], length: int, expected: list[str]) -> None:
    """Return the values below 2**length that are absent from the list.

    Args:
        lst: Known `bin()` strings.
        length: Number of bits to generate.
        expected: Expected missing `bin()` strings.
    """
    assert getLackingBinaries(lst, length=length) == expected


@pytest.mark.parametrize(
    ('lst', 'expected'),
    [
        (['0b11'], ['0b0', '0b1', '0b10']),
        (
            ['0b101'],
            ['0b0', '0b1', '0b10', '0b11', '0b100', '0b110', '0b111'],
        ),
        (['0b1', '0b101'], ['0b0', '0b10', '0b11', '0b100', '0b110', '0b111']),
    ],
)
def test_default_length_is_longest_entry_without_prefix(
    lst: list[str], expected: list[str]
) -> None:
    """Derive the number of bits from the longest entry minus the `0b` prefix.

    Args:
        lst: Known `bin()` strings.
        expected: Expected missing `bin()` strings.
    """
    assert getLackingBinaries(lst) == expected


def test_entries_outside_the_generated_range_are_ignored() -> None:
    """Ignore known entries that are larger than 2**length - 1."""
    assert getLackingBinaries(['0b1111'], length=2) == ['0b0', '0b1', '0b10', '0b11']


@pytest.mark.parametrize('known', ['1', '0b01', '0B1', ' 0b1', '0b1 '])
def test_entries_not_in_bin_format_never_match(known: str) -> None:
    """Keep a value missing when its known entry is not the exact `bin()` string.

    Args:
        known: Entry that looks like 1 but differs from `bin(1)`.
    """
    assert getLackingBinaries([known], length=1) == ['0b0', '0b1']


@pytest.mark.parametrize('seed', range(5))
def test_result_and_known_entries_partition_the_whole_range(seed: int) -> None:
    """Split the range 0..2**length - 1 between the known entries and the result.

    Args:
        seed: Seed of the random generator that picks the known entries.
    """
    length = 4
    rng = np.random.default_rng(seed)
    known = rng.choice(2**length, size=rng.integers(0, 2**length + 1), replace=False)
    lst = [bin(int(value)) for value in known]

    result = getLackingBinaries(lst, length=length)

    universe = {bin(value) for value in range(2**length)}
    assert set(result).isdisjoint(lst)
    assert set(result) | set(lst) == universe
    assert len(result) == 2**length - len(lst)


def test_result_is_in_ascending_numeric_order() -> None:
    """Return the strings sorted by the integer they represent."""
    result = getLackingBinaries(['0b101'], length=3)

    values = [int(binary, 2) for binary in result]
    assert values == sorted(values)


def test_empty_list_without_length_raises_value_error() -> None:
    """Raise ValueError when no length can be derived from an empty list."""
    with pytest.raises(ValueError):
        getLackingBinaries([])
