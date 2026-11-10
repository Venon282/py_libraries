"""Unit tests for UnitConverter.isSupported."""

import pytest

from py_libraries.other.UnitConverter import UnitConverter

CONVERTER = UnitConverter()


@pytest.mark.parametrize(
    ('unit', 'dimension'),
    [
        ('metre', 'length'),
        ('kilometre', 'length'),
        ('kilogram', 'mass'),
        ('square_foot', 'area'),
        ('per_square_angstrom', 'inverse_area'),
        ('celsius', 'temperature'),
        ('kelvin', 'temperature'),
        ('mmhg', 'pressure'),
    ],
)
def test_supported_units(unit: str, dimension: str) -> None:
    """Accept a unit of the dimension.

    Args:
        unit: Unit name.
        dimension: Dimension that holds the unit.
    """
    assert CONVERTER.isSupported(unit, dimension) is True


@pytest.mark.parametrize(
    ('unit', 'dimension'),
    [
        ('parsec', 'length'),
        ('kilogram', 'length'),
        ('metre', 'mass'),
        ('metre', 'temperature'),
        ('celsius', 'length'),
        ('', 'length'),
    ],
)
def test_unsupported_units(unit: str, dimension: str) -> None:
    """Reject a unit that is unknown or belongs to another dimension.

    Args:
        unit: Unit name.
        dimension: Dimension that does not hold the unit.
    """
    assert CONVERTER.isSupported(unit, dimension) is False


@pytest.mark.parametrize('dimension', ['colour', '', 'lengths'])
def test_unknown_dimension_gives_false_instead_of_raising(dimension: str) -> None:
    """Return False for an unknown dimension.

    Args:
        dimension: Name that is not a dimension.
    """
    assert CONVERTER.isSupported('metre', dimension) is False


@pytest.mark.parametrize(
    ('unit', 'dimension'),
    [
        ('METRE', 'length'),
        ('Metre', 'LENGTH'),
        ('mmHg', 'pressure'),
        ('CELSIUS', 'Temperature'),
    ],
)
def test_names_ignore_case(unit: str, dimension: str) -> None:
    """Ignore the case of the unit and of the dimension.

    Args:
        unit: Unit name, in any case.
        dimension: Dimension, in any case.
    """
    assert CONVERTER.isSupported(unit, dimension) is True


def test_agrees_with_listUnits_for_every_dimension() -> None:
    """Accept exactly the units that `listUnits` returns."""
    for dimension in CONVERTER.listDimensions():
        listed = CONVERTER.listUnits(dimension)
        others = {
            unit
            for other in CONVERTER.listDimensions()
            if other != dimension
            for unit in CONVERTER.listUnits(other)
        } - set(listed)

        assert all(CONVERTER.isSupported(unit, dimension) for unit in listed)
        assert not any(CONVERTER.isSupported(unit, dimension) for unit in others)
