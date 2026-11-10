"""Unit tests for UnitConverter.convert."""

import math

import pytest

from py_libraries.other.UnitConverter import UnitConverter

CONVERTER = UnitConverter()
DIMENSIONS = CONVERTER.listDimensions()
LINEAR_DIMENSIONS = [dim for dim in DIMENSIONS if dim != 'temperature']

# Exact relations between two units of the same dimension, taken from their
# definitions rather than from the factor tables.
EXACT_RELATIONS = [
    # length
    (1, 'kilometre', 'metre', 'length', 1_000),
    (1, 'inch', 'centimetre', 'length', 2.54),
    (1, 'foot', 'inch', 'length', 12),
    (1, 'yard', 'foot', 'length', 3),
    (1, 'mile', 'foot', 'length', 5_280),
    (1, 'nautical_mile', 'metre', 'length', 1_852),
    (1, 'angstrom', 'nanometre', 'length', 0.1),
    # mass
    (1, 'tonne', 'kilogram', 'mass', 1_000),
    (1, 'pound', 'ounce', 'mass', 16),
    (1, 'stone', 'pound', 'mass', 14),
    (1, 'short_ton', 'pound', 'mass', 2_000),
    (1, 'long_ton', 'pound', 'mass', 2_240),
    (1, 'pound', 'gram', 'mass', 453.592_37),
    # time
    (1, 'minute', 'second', 'time', 60),
    (1, 'hour', 'minute', 'time', 60),
    (1, 'day', 'hour', 'time', 24),
    (1, 'week', 'day', 'time', 7),
    (1, 'year', 'day', 'time', 365.25),
    # area
    (1, 'hectare', 'square_metre', 'area', 10_000),
    (1, 'acre', 'square_foot', 'area', 43_560),
    (1, 'square_mile', 'acre', 'area', 640),
    # volume
    (1, 'litre', 'cubic_centimetre', 'volume', 1_000),
    (1, 'litre', 'millilitre', 'volume', 1_000),
    (1, 'gallon_us', 'quart_us', 'volume', 4),
    (1, 'quart_us', 'pint_us', 'volume', 2),
    (1, 'pint_us', 'fluid_ounce_us', 'volume', 16),
    (1, 'gallon_uk', 'pint_uk', 'volume', 8),
    (1, 'pint_uk', 'fluid_ounce_uk', 'volume', 20),
    (1, 'tablespoon', 'teaspoon', 'volume', 3),
    (1, 'gallon_us', 'litre', 'volume', 3.785_411_784),
    (1, 'gallon_uk', 'litre', 'volume', 4.546_09),
    # speed
    (1, 'knot', 'kilometre_per_hour', 'speed', 1.852),
    (1, 'mile_per_hour', 'foot_per_second', 'speed', 5_280 / 3_600),
    (1, 'metre_per_second', 'kilometre_per_hour', 'speed', 3.6),
    # pressure
    (1, 'bar', 'pascal', 'pressure', 100_000),
    (1, 'bar', 'millibar', 'pressure', 1_000),
    (1, 'atmosphere', 'pascal', 'pressure', 101_325),
    (1, 'atmosphere', 'torr', 'pressure', 760),
    # energy
    (1, 'kilocalorie', 'calorie', 'energy', 1_000),
    (1, 'watt_hour', 'joule', 'energy', 3_600),
    (1, 'kilowatt_hour', 'megajoule', 'energy', 3.6),
    # power
    (1, 'kilowatt', 'watt', 'power', 1_000),
    (1, 'horsepower', 'foot_pound_per_second', 'power', 550),
    # data
    (1, 'byte', 'bit', 'data', 8),
    (1, 'nibble', 'bit', 'data', 4),
    (1, 'kilobyte', 'byte', 'data', 1_000),
    (1, 'megabyte', 'kilobyte', 'data', 1_000),
    (1, 'kibibyte', 'byte', 'data', 1_024),
    (1, 'gibibyte', 'mebibyte', 'data', 1_024),
    # angle
    (180, 'degree', 'radian', 'angle', math.pi),
    (1, 'turn', 'degree', 'angle', 360),
    (1, 'turn', 'gradian', 'angle', 400),
    (1, 'degree', 'arcminute', 'angle', 60),
    (1, 'arcminute', 'arcsecond', 'angle', 60),
]


@pytest.mark.parametrize(
    ('value', 'from_unit', 'to_unit', 'dimension', 'expected'), EXACT_RELATIONS
)
def test_known_relations(
    value: float, from_unit: str, to_unit: str, dimension: str, expected: float
) -> None:
    """Convert units whose ratio comes from their definition.

    Args:
        value: Value to convert.
        from_unit: Source unit.
        to_unit: Target unit.
        dimension: Dimension of both units.
        expected: Expected converted value.
    """
    result = CONVERTER.convert(value, from_unit, to_unit, dimension)

    assert result == pytest.approx(expected, rel=1e-9)


def test_hundred_kilometres_in_miles() -> None:
    """Convert the example of the documentation."""
    result = CONVERTER.convert(100, 'kilometre', 'mile', 'length')

    assert result == pytest.approx(62.137_119_223_733, rel=1e-9)


@pytest.mark.parametrize(
    ('area_unit', 'length_unit'),
    [
        ('square_kilometre', 'kilometre'),
        ('square_centimetre', 'centimetre'),
        ('square_millimetre', 'millimetre'),
        ('square_inch', 'inch'),
        ('square_foot', 'foot'),
        ('square_yard', 'yard'),
        ('square_mile', 'mile'),
    ],
)
def test_area_unit_is_the_square_of_its_length_unit(
    area_unit: str, length_unit: str
) -> None:
    """Give an area unit equal to the square of the matching length unit.

    Args:
        area_unit: Area unit.
        length_unit: Length unit it is built from.
    """
    side = CONVERTER.convert(1, length_unit, 'metre', 'length')

    area = CONVERTER.convert(1, area_unit, 'square_metre', 'area')

    assert area == pytest.approx(side**2, rel=1e-9)


@pytest.mark.parametrize(
    ('inverse_unit', 'length_unit'),
    [
        ('per_square_kilometre', 'kilometre'),
        ('per_square_centimetre', 'centimetre'),
        ('per_square_millimetre', 'millimetre'),
        ('per_square_micrometre', 'micrometre'),
        ('per_square_nanometre', 'nanometre'),
        ('per_square_angstrom', 'angstrom'),
        ('per_square_inch', 'inch'),
        ('per_square_foot', 'foot'),
        ('per_square_yard', 'yard'),
        ('per_square_mile', 'mile'),
    ],
)
def test_inverse_area_unit_is_the_inverse_square_of_its_length_unit(
    inverse_unit: str, length_unit: str
) -> None:
    """Give an inverse area unit equal to one over the squared length unit.

    Args:
        inverse_unit: Inverse area unit.
        length_unit: Length unit it is built from.
    """
    side = CONVERTER.convert(1, length_unit, 'metre', 'length')

    density = CONVERTER.convert(1, inverse_unit, 'per_square_metre', 'inverse_area')

    assert density == pytest.approx(1 / side**2, rel=1e-9)


@pytest.mark.parametrize(
    ('volume_unit', 'length_unit'),
    [
        ('cubic_centimetre', 'centimetre'),
        ('cubic_inch', 'inch'),
        ('cubic_foot', 'foot'),
        ('cubic_yard', 'yard'),
        ('angstrom_cubed', 'angstrom'),
    ],
)
def test_volume_unit_is_the_cube_of_its_length_unit(
    volume_unit: str, length_unit: str
) -> None:
    """Give a volume unit equal to the cube of the matching length unit.

    Args:
        volume_unit: Volume unit.
        length_unit: Length unit it is built from.
    """
    side = CONVERTER.convert(1, length_unit, 'metre', 'length')

    volume = CONVERTER.convert(1, volume_unit, 'cubic_metre', 'volume')

    assert volume == pytest.approx(side**3, rel=1e-9)


@pytest.mark.parametrize(
    ('speed_unit', 'length_unit', 'time_unit'),
    [
        ('kilometre_per_hour', 'kilometre', 'hour'),
        ('mile_per_hour', 'mile', 'hour'),
        ('foot_per_second', 'foot', 'second'),
        ('knot', 'nautical_mile', 'hour'),
        ('metre_per_second', 'metre', 'second'),
    ],
)
def test_speed_unit_is_a_length_unit_per_time_unit(
    speed_unit: str, length_unit: str, time_unit: str
) -> None:
    """Give a speed unit equal to its length unit divided by its time unit.

    Args:
        speed_unit: Speed unit.
        length_unit: Length unit it is built from.
        time_unit: Time unit it is built from.
    """
    length = CONVERTER.convert(1, length_unit, 'metre', 'length')
    duration = CONVERTER.convert(1, time_unit, 'second', 'time')

    speed = CONVERTER.convert(1, speed_unit, 'metre_per_second', 'speed')

    assert speed == pytest.approx(length / duration, rel=1e-9)


@pytest.mark.parametrize(
    ('value', 'from_unit', 'to_unit', 'expected'),
    [
        (0, 'celsius', 'fahrenheit', 32),
        (100, 'celsius', 'fahrenheit', 212),
        (-40, 'celsius', 'fahrenheit', -40),
        (212, 'fahrenheit', 'celsius', 100),
        (0, 'celsius', 'kelvin', 273.15),
        (0, 'kelvin', 'celsius', -273.15),
        (0, 'kelvin', 'rankine', 0),
        (0, 'kelvin', 'fahrenheit', -459.67),
        (-459.67, 'fahrenheit', 'kelvin', 0),
        (0, 'celsius', 'rankine', 491.67),
        (32, 'fahrenheit', 'rankine', 491.67),
        (491.67, 'rankine', 'celsius', 0),
    ],
)
def test_temperature_known_values(
    value: float, from_unit: str, to_unit: str, expected: float
) -> None:
    """Convert the fixed points of the temperature scales.

    Args:
        value: Temperature to convert.
        from_unit: Source scale.
        to_unit: Target scale.
        expected: Expected temperature.
    """
    result = CONVERTER.convert(value, from_unit, to_unit, 'temperature')

    assert result == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize('celsius', [-273.15, -40.0, -1.5, 0.0, 21.5, 100.0, 1_000.0])
def test_temperature_follows_the_fahrenheit_formula(celsius: float) -> None:
    """Match F = C * 9 / 5 + 32.

    Args:
        celsius: Temperature in degrees Celsius.
    """
    result = CONVERTER.convert(celsius, 'celsius', 'fahrenheit', 'temperature')

    assert result == pytest.approx(celsius * 9 / 5 + 32, abs=1e-9)


def test_temperature_difference_scales_between_kelvin_and_rankine() -> None:
    """Keep the ratio 9 / 5 between the kelvin and the Rankine degrees."""
    low = CONVERTER.convert(10, 'kelvin', 'rankine', 'temperature')
    high = CONVERTER.convert(20, 'kelvin', 'rankine', 'temperature')

    assert high - low == pytest.approx(10 * 9 / 5)


def test_temperature_below_absolute_zero_is_not_rejected() -> None:
    """Convert a temperature below the absolute zero like any other."""
    result = CONVERTER.convert(-300, 'celsius', 'kelvin', 'temperature')

    assert result == pytest.approx(-26.85)


@pytest.mark.parametrize('dimension', DIMENSIONS)
def test_converting_to_the_same_unit_keeps_the_value(dimension: str) -> None:
    """Return the input value when the source and target units are the same.

    Args:
        dimension: Dimension whose units are all tested.
    """
    for unit in CONVERTER.listUnits(dimension):
        assert CONVERTER.convert(12.5, unit, unit, dimension) == pytest.approx(12.5)


@pytest.mark.parametrize('dimension', DIMENSIONS)
def test_round_trip_restores_the_value(dimension: str) -> None:
    """Give back the value when converting to another unit and back.

    Args:
        dimension: Dimension whose pairs of units are all tested.
    """
    units = CONVERTER.listUnits(dimension)

    for from_unit in units:
        for to_unit in units:
            there = CONVERTER.convert(25.0, from_unit, to_unit, dimension)
            back = CONVERTER.convert(there, to_unit, from_unit, dimension)

            assert back == pytest.approx(25.0, rel=1e-9)


@pytest.mark.parametrize('dimension', DIMENSIONS)
def test_converting_through_an_intermediate_unit_is_consistent(
    dimension: str,
) -> None:
    """Give the same result directly or through a third unit.

    Args:
        dimension: Dimension whose units are tested.
    """
    units = CONVERTER.listUnits(dimension)
    first, middle, last = units[0], units[len(units) // 2], units[-1]

    direct = CONVERTER.convert(7.0, first, last, dimension)
    indirect = CONVERTER.convert(
        CONVERTER.convert(7.0, first, middle, dimension), middle, last, dimension
    )

    assert indirect == pytest.approx(direct, rel=1e-9)


@pytest.mark.parametrize('dimension', LINEAR_DIMENSIONS)
def test_conversion_is_proportional_to_the_value(dimension: str) -> None:
    """Scale the result with the value, and map zero to zero.

    Args:
        dimension: Dimension other than the temperature.
    """
    units = CONVERTER.listUnits(dimension)
    from_unit, to_unit = units[0], units[-1]

    once = CONVERTER.convert(2.0, from_unit, to_unit, dimension)
    triple = CONVERTER.convert(6.0, from_unit, to_unit, dimension)

    assert triple == pytest.approx(3 * once, rel=1e-12)
    assert CONVERTER.convert(0.0, from_unit, to_unit, dimension) == 0.0


@pytest.mark.parametrize('dimension', LINEAR_DIMENSIONS)
def test_negative_value_gives_the_opposite_result(dimension: str) -> None:
    """Keep the sign of the value.

    Args:
        dimension: Dimension other than the temperature.
    """
    units = CONVERTER.listUnits(dimension)
    from_unit, to_unit = units[0], units[-1]

    positive = CONVERTER.convert(4.0, from_unit, to_unit, dimension)

    assert CONVERTER.convert(-4.0, from_unit, to_unit, dimension) == -positive


def test_unit_and_dimension_names_ignore_case() -> None:
    """Accept any case for the units and the dimension."""
    expected = CONVERTER.convert(3, 'kilometre', 'mile', 'length')

    assert CONVERTER.convert(3, 'KILOMETRE', 'Mile', 'LENGTH') == expected
    assert CONVERTER.convert(3, 'mmHg', 'PSI', 'Pressure') == pytest.approx(
        CONVERTER.convert(3, 'mmhg', 'psi', 'pressure')
    )
    assert CONVERTER.convert(10, 'Celsius', 'KELVIN', 'Temperature') == pytest.approx(
        283.15
    )


def test_integer_value_gives_a_float() -> None:
    """Return a float for an integer input."""
    assert isinstance(CONVERTER.convert(3, 'metre', 'foot', 'length'), float)


def test_unknown_dimension_raises_value_error() -> None:
    """Raise ValueError and list the supported dimensions."""
    with pytest.raises(ValueError, match="Unknown dimension 'colour'") as excinfo:
        CONVERTER.convert(1, 'metre', 'foot', 'colour')

    assert 'length' in str(excinfo.value)
    assert 'temperature' in str(excinfo.value)


@pytest.mark.parametrize(
    ('from_unit', 'to_unit', 'unknown'),
    [
        ('parsec', 'metre', 'parsec'),
        ('metre', 'parsec', 'parsec'),
        ('kilogram', 'metre', 'kilogram'),
        ('metre', 'kilogram', 'kilogram'),
        ('', 'metre', ''),
    ],
)
def test_unknown_unit_raises_value_error(
    from_unit: str, to_unit: str, unknown: str
) -> None:
    """Raise ValueError that names the unit missing from the dimension.

    Args:
        from_unit: Source unit.
        to_unit: Target unit.
        unknown: Unit that does not belong to the length dimension.
    """
    with pytest.raises(ValueError, match=f'Unknown unit {unknown!r}'):
        CONVERTER.convert(1, from_unit, to_unit, 'length')


def test_unknown_unit_message_keeps_the_original_case() -> None:
    """Quote the unit as the caller wrote it, not in lower case."""
    with pytest.raises(ValueError, match="Unknown unit 'Parsec'"):
        CONVERTER.convert(1, 'Parsec', 'metre', 'length')


@pytest.mark.parametrize(
    ('from_unit', 'to_unit', 'unknown'),
    [
        ('metre', 'kelvin', 'metre'),
        ('kelvin', 'metre', 'metre'),
        ('Reaumur', 'kelvin', 'Reaumur'),
    ],
)
def test_unknown_temperature_unit_raises_value_error(
    from_unit: str, to_unit: str, unknown: str
) -> None:
    """Raise ValueError when a unit is not a temperature scale.

    Args:
        from_unit: Source scale.
        to_unit: Target scale.
        unknown: Unit that is not a temperature scale.
    """
    with pytest.raises(ValueError, match=f'Unknown temperature unit {unknown!r}'):
        CONVERTER.convert(1, from_unit, to_unit, 'temperature')
