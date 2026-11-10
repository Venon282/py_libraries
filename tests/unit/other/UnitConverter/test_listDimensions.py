"""Unit tests for UnitConverter.listDimensions."""

import pytest

from py_libraries.other.UnitConverter import UnitConverter

EXPECTED_DIMENSIONS = [
    'angle',
    'area',
    'data',
    'energy',
    'inverse_area',
    'length',
    'mass',
    'power',
    'pressure',
    'speed',
    'temperature',
    'time',
    'volume',
]


def test_lists_every_supported_dimension_sorted() -> None:
    """Return the thirteen dimensions in alphabetical order."""
    assert UnitConverter().listDimensions() == EXPECTED_DIMENSIONS


def test_includes_the_temperature_that_has_no_factor_table() -> None:
    """Add the temperature to the dimensions that have a factor table."""
    assert 'temperature' in UnitConverter().listDimensions()


def test_every_dimension_can_list_its_units() -> None:
    """Accept each returned dimension in `listUnits`."""
    converter = UnitConverter()

    for dimension in converter.listDimensions():
        assert converter.listUnits(dimension)


def test_result_is_a_new_list_on_each_call() -> None:
    """Keep the internal state intact when the caller edits the result."""
    converter = UnitConverter()
    first = converter.listDimensions()

    first.append('colour')
    first.clear()

    assert converter.listDimensions() == EXPECTED_DIMENSIONS


@pytest.mark.parametrize('dimension', EXPECTED_DIMENSIONS)
def test_dimension_names_are_lower_case(dimension: str) -> None:
    """Use lower case names, as `convert` lowers the name it receives.

    Args:
        dimension: Expected dimension name.
    """
    assert dimension == dimension.lower()
