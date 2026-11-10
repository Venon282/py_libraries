"""Unit tests for UnitConverter.listUnits."""

import pytest

from py_libraries.other.UnitConverter import UnitConverter

CONVERTER = UnitConverter()
DIMENSIONS = CONVERTER.listDimensions()


def test_pressure_units() -> None:
    """List every pressure unit, sorted."""
    assert CONVERTER.listUnits('pressure') == [
        'atmosphere',
        'bar',
        'kilopascal',
        'megapascal',
        'millibar',
        'mmhg',
        'pascal',
        'psi',
        'torr',
    ]


def test_temperature_units() -> None:
    """List the four temperature scales, sorted."""
    assert CONVERTER.listUnits('temperature') == [
        'celsius',
        'fahrenheit',
        'kelvin',
        'rankine',
    ]


def test_angle_units() -> None:
    """List the angle units, sorted."""
    assert CONVERTER.listUnits('angle') == [
        'arcminute',
        'arcsecond',
        'degree',
        'gradian',
        'radian',
        'turn',
    ]


@pytest.mark.parametrize('dimension', DIMENSIONS)
def test_units_are_sorted_unique_and_not_empty(dimension: str) -> None:
    """Return a sorted list without duplicate.

    Args:
        dimension: Dimension to list.
    """
    units = CONVERTER.listUnits(dimension)

    assert units
    assert units == sorted(set(units))


@pytest.mark.parametrize('dimension', DIMENSIONS)
def test_unit_names_are_lower_case(dimension: str) -> None:
    """Use lower case names, as `convert` lowers the unit names it receives.

    Args:
        dimension: Dimension to list.
    """
    for unit in CONVERTER.listUnits(dimension):
        assert unit == unit.lower()


@pytest.mark.parametrize('dimension', DIMENSIONS)
def test_every_listed_unit_can_be_converted(dimension: str) -> None:
    """Accept each listed unit as a source and as a target of `convert`.

    Args:
        dimension: Dimension to list.
    """
    for unit in CONVERTER.listUnits(dimension):
        CONVERTER.convert(1, unit, unit, dimension)


@pytest.mark.parametrize('spelling', ['length', 'LENGTH', 'Length', 'lEnGtH'])
def test_dimension_name_ignores_case(spelling: str) -> None:
    """Return the same units whatever the case of the dimension.

    Args:
        spelling: Spelling of the dimension.
    """
    assert CONVERTER.listUnits(spelling) == CONVERTER.listUnits('length')


def test_temperature_name_ignores_case() -> None:
    """Return the temperature scales whatever the case of the dimension."""
    assert CONVERTER.listUnits('TEMPERATURE') == CONVERTER.listUnits('temperature')


@pytest.mark.parametrize('dimension', ['length', 'temperature'])
def test_result_is_a_new_list_on_each_call(dimension: str) -> None:
    """Keep the internal state intact when the caller edits the result.

    Args:
        dimension: Dimension to list.
    """
    expected = CONVERTER.listUnits(dimension)
    result = CONVERTER.listUnits(dimension)

    result.append('parsec')
    result.clear()

    assert CONVERTER.listUnits(dimension) == expected


@pytest.mark.parametrize('dimension', ['colour', '', 'lengths'])
def test_unknown_dimension_raises_value_error(dimension: str) -> None:
    """Raise ValueError and list the supported dimensions.

    Args:
        dimension: Name that is not a dimension.
    """
    with pytest.raises(ValueError, match='Unknown dimension') as excinfo:
        CONVERTER.listUnits(dimension)

    assert 'length' in str(excinfo.value)
