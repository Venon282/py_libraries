"""Convert numeric values between units of the same physical dimension."""

import math
from collections.abc import Callable, Mapping
from types import MappingProxyType

# Each factor table maps a unit name to the value of that unit expressed in the SI
# base unit of its dimension, given in the first comment of the table. The
# temperature is not in a table because its scales are linked by offsets.

_LENGTH_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        # SI base: metre
        'metre': 1.0,
        'kilometre': 1_000.0,
        'centimetre': 1e-2,
        'millimetre': 1e-3,
        'micrometre': 1e-6,
        'nanometre': 1e-9,
        'angstrom': 1e-10,
        'inch': 0.0254,
        'foot': 0.3048,
        'yard': 0.9144,
        'mile': 1_609.344,
        'nautical_mile': 1_852.0,
        'light_year': 9.460_730_472_580_8e15,
    }
)

_MASS_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        # SI base: kilogram
        'kilogram': 1.0,
        'gram': 1e-3,
        'milligram': 1e-6,
        'microgram': 1e-9,
        'tonne': 1_000.0,
        'pound': 0.453_592_37,
        'ounce': 0.028_349_523_125,
        'stone': 6.350_293_18,
        'short_ton': 907.184_74,
        'long_ton': 1_016.046_9088,
    }
)

_TIME_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        # SI base: second
        'second': 1.0,
        'millisecond': 1e-3,
        'microsecond': 1e-6,
        'nanosecond': 1e-9,
        'minute': 60.0,
        'hour': 3_600.0,
        'day': 86_400.0,
        'week': 604_800.0,
        'year': 31_557_600.0,  # Julian year
    }
)

_AREA_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        # SI base: square metre
        'square_metre': 1.0,
        'square_kilometre': 1e6,
        'square_centimetre': 1e-4,
        'square_millimetre': 1e-6,
        'square_inch': 6.4516e-4,
        'square_foot': 0.092_903_04,
        'square_yard': 0.836_127_36,
        'square_mile': 2_589_988.110_336,
        'acre': 4_046.856_422_4,
        'hectare': 10_000.0,
    }
)

_INVERSE_AREA_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        # SI base: per square metre (m^-2)
        # Factor = how many m^-2 equals 1 of this unit
        'per_square_metre': 1.0,
        'per_square_kilometre': 1e-6,
        'per_square_centimetre': 1e4,
        'per_square_millimetre': 1e6,
        'per_square_micrometre': 1e12,
        'per_square_nanometre': 1e18,
        'per_square_angstrom': 1e20,
        'per_square_inch': 1.0 / 6.4516e-4,
        'per_square_foot': 1.0 / 0.092_903_04,
        'per_square_yard': 1.0 / 0.836_127_36,
        'per_square_mile': 1.0 / 2_589_988.110_336,
    }
)

_VOLUME_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        # SI base: cubic metre
        'cubic_metre': 1.0,
        'litre': 1e-3,
        'millilitre': 1e-6,
        'centilitre': 1e-5,
        'cubic_centimetre': 1e-6,
        'angstrom_cubed': 1e-30,
        'cubic_inch': 1.638_706_4e-5,
        'cubic_foot': 0.028_316_846_592,
        'cubic_yard': 0.764_554_857_984,
        'gallon_us': 3.785_411_784e-3,
        'gallon_uk': 4.546_09e-3,
        'quart_us': 9.463_529_46e-4,
        'pint_us': 4.731_764_73e-4,
        'pint_uk': 5.682_612_5e-4,
        'fluid_ounce_us': 2.957_352_956_25e-5,
        'fluid_ounce_uk': 2.841_306_25e-5,
        'tablespoon': 1.478_676_478_125e-5,
        'teaspoon': 4.928_921_593_75e-6,
    }
)

_SPEED_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        # SI base: metre per second
        'metre_per_second': 1.0,
        'kilometre_per_hour': 1.0 / 3.6,
        'mile_per_hour': 0.447_04,
        'foot_per_second': 0.3048,
        # Defined as one nautical mile (1852 m) per hour
        'knot': 1_852.0 / 3_600.0,
    }
)

_PRESSURE_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        # SI base: pascal
        'pascal': 1.0,
        'kilopascal': 1_000.0,
        'megapascal': 1_000_000.0,
        'bar': 100_000.0,
        'millibar': 100.0,
        'atmosphere': 101_325.0,
        'torr': 133.322_368_421,
        'mmhg': 133.322_387_415,
        'psi': 6_894.757_293_168,
    }
)

_ENERGY_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        # SI base: joule
        'joule': 1.0,
        'kilojoule': 1_000.0,
        'megajoule': 1_000_000.0,
        'calorie': 4.184,
        'kilocalorie': 4_184.0,
        'watt_hour': 3_600.0,
        'kilowatt_hour': 3_600_000.0,
        'electron_volt': 1.602_176_634e-19,
        'btu': 1_055.056,
        'foot_pound': 1.355_817_948_331_4,
        'erg': 1e-7,
    }
)

_POWER_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        # SI base: watt
        'watt': 1.0,
        'kilowatt': 1_000.0,
        'megawatt': 1_000_000.0,
        'milliwatt': 1e-3,
        'horsepower': 745.699_871_582_270_22,
        'btu_per_hour': 0.293_071,
        'foot_pound_per_second': 1.355_817_948_331_4,
    }
)

_DATA_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        # SI base: bit
        'bit': 1.0,
        'nibble': 4.0,
        'byte': 8.0,
        'kilobit': 1_000.0,
        'kilobyte': 8_000.0,
        'megabit': 1_000_000.0,
        'megabyte': 8_000_000.0,
        'gigabit': 1e9,
        'gigabyte': 8e9,
        'terabit': 1e12,
        'terabyte': 8e12,
        'petabit': 1e15,
        'petabyte': 8e15,
        'kibibyte': 8.0 * 1_024,
        'mebibyte': 8.0 * 1_024**2,
        'gibibyte': 8.0 * 1_024**3,
        'tebibyte': 8.0 * 1_024**4,
    }
)

_RADIANS_PER_DEGREE = math.pi / 180.0

_ANGLE_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        # SI base: radian
        'radian': 1.0,
        'degree': _RADIANS_PER_DEGREE,
        'gradian': math.pi / 200.0,
        'arcminute': _RADIANS_PER_DEGREE / 60.0,
        'arcsecond': _RADIANS_PER_DEGREE / 3_600.0,
        'turn': 2.0 * math.pi,
    }
)

# Map each dimension name to its factor table.
_DIMENSION_MAP: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        'length': _LENGTH_FACTORS,
        'mass': _MASS_FACTORS,
        'time': _TIME_FACTORS,
        'area': _AREA_FACTORS,
        'inverse_area': _INVERSE_AREA_FACTORS,
        'volume': _VOLUME_FACTORS,
        'speed': _SPEED_FACTORS,
        'pressure': _PRESSURE_FACTORS,
        'energy': _ENERGY_FACTORS,
        'power': _POWER_FACTORS,
        'data': _DATA_FACTORS,
        'angle': _ANGLE_FACTORS,
    }
)

# The temperature scales are linked by offsets, which a multiplicative factor cannot
# express, so each scale is converted to and from the kelvin by a function. The
# kelvin is the canonical base: it avoids one formula per pair of scales.
_TEMPERATURE = 'temperature'

_KELVIN_OFFSET_OF_CELSIUS = 273.15
_RANKINE_OFFSET_OF_FAHRENHEIT = 459.67
_RANKINE_PER_KELVIN = 9.0 / 5.0

_TO_KELVIN: Mapping[str, Callable[[float], float]] = MappingProxyType(
    {
        'kelvin': lambda value: value,
        'celsius': lambda value: value + _KELVIN_OFFSET_OF_CELSIUS,
        'fahrenheit': lambda value: (
            (value + _RANKINE_OFFSET_OF_FAHRENHEIT) / _RANKINE_PER_KELVIN
        ),
        'rankine': lambda value: value / _RANKINE_PER_KELVIN,
    }
)

_FROM_KELVIN: Mapping[str, Callable[[float], float]] = MappingProxyType(
    {
        'kelvin': lambda value: value,
        'celsius': lambda value: value - _KELVIN_OFFSET_OF_CELSIUS,
        'fahrenheit': lambda value: (
            value * _RANKINE_PER_KELVIN - _RANKINE_OFFSET_OF_FAHRENHEIT
        ),
        'rankine': lambda value: value * _RANKINE_PER_KELVIN,
    }
)


def _convertTemperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a temperature through the kelvin.

    Args:
        value: Temperature to convert.
        from_unit: Source temperature scale (case-insensitive).
        to_unit: Target temperature scale (case-insensitive).

    Returns:
        The temperature expressed in `to_unit`.

    Raises:
        ValueError: If `from_unit` or `to_unit` is not a temperature scale.
    """
    to_kelvin = _TO_KELVIN.get(from_unit.lower())
    if to_kelvin is None:
        raise ValueError(f'Unknown temperature unit {from_unit!r}.')

    from_kelvin = _FROM_KELVIN.get(to_unit.lower())
    if from_kelvin is None:
        raise ValueError(f'Unknown temperature unit {to_unit!r}.')

    return from_kelvin(to_kelvin(value))


class UnitConverter:
    """Convert numeric values between units within a given physical dimension.

    The supported dimensions are length, mass, time, temperature, area,
    inverse_area, volume, speed, pressure, energy, power, data and angle. The unit
    and dimension names are case-insensitive, and the units are written in lower
    case with underscores (`'square_foot'`, `'mile_per_hour'`). Use `listUnits` to
    get the units of a dimension.

    Examples:
        >>> converter = UnitConverter()
        >>> round(converter.convert(100, 'kilometre', 'mile', 'length'), 4)
        62.1371
        >>> round(converter.convert(100, 'celsius', 'fahrenheit', 'temperature'), 6)
        212.0
        >>> converter.listUnits('temperature')
        ['celsius', 'fahrenheit', 'kelvin', 'rankine']
    """

    def convert(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        dimension: str,
    ) -> float:
        """Convert a value from one unit to another within a dimension.

        The value is not range checked: a temperature below the absolute zero is
        converted like any other.

        Args:
            value: Numeric value to convert.
            from_unit: Source unit name (case-insensitive).
            to_unit: Target unit name (case-insensitive).
            dimension: Physical dimension of both units (case-insensitive).

        Returns:
            The value expressed in `to_unit`.

        Raises:
            ValueError: If the dimension is unknown, or if a unit does not belong to
                the dimension.
        """
        if dimension.lower() == _TEMPERATURE:
            return _convertTemperature(value, from_unit, to_unit)

        factor_table = self._getFactorTable(dimension)
        from_factor = self._getFactor(factor_table, from_unit, dimension)
        to_factor = self._getFactor(factor_table, to_unit, dimension)

        return value * from_factor / to_factor

    def listDimensions(self) -> list[str]:
        """List the supported dimensions.

        Returns:
            The dimension names, sorted alphabetically.
        """
        return sorted([*_DIMENSION_MAP, _TEMPERATURE])

    def listUnits(self, dimension: str) -> list[str]:
        """List the units available for a dimension.

        Args:
            dimension: Physical dimension (case-insensitive).

        Returns:
            The unit names of `dimension`, sorted alphabetically.

        Raises:
            ValueError: If the dimension is unknown.
        """
        if dimension.lower() == _TEMPERATURE:
            return sorted(_TO_KELVIN)

        return sorted(self._getFactorTable(dimension))

    def isSupported(self, unit: str, dimension: str) -> bool:
        """Check whether a unit is supported within a dimension.

        Args:
            unit: Unit name (case-insensitive).
            dimension: Physical dimension (case-insensitive).

        Returns:
            True if `unit` belongs to `dimension`. False if it does not, or if the
            dimension is unknown.
        """
        try:
            return unit.lower() in self.listUnits(dimension)
        except ValueError:
            return False

    def _getFactorTable(self, dimension: str) -> Mapping[str, float]:
        """Return the factor table of a dimension other than the temperature.

        Args:
            dimension: Physical dimension (case-insensitive).

        Returns:
            The mapping from the unit names to their factor in the SI base unit.

        Raises:
            ValueError: If the dimension is unknown.
        """
        factor_table = _DIMENSION_MAP.get(dimension.lower())
        if factor_table is None:
            supported = ', '.join(self.listDimensions())
            raise ValueError(
                f'Unknown dimension {dimension!r}, expected one of: {supported}.'
            )
        return factor_table

    @staticmethod
    def _getFactor(
        factor_table: Mapping[str, float], unit: str, dimension: str
    ) -> float:
        """Return the factor of a unit in the SI base unit of its dimension.

        Args:
            factor_table: Factor table of the dimension.
            unit: Unit name (case-insensitive).
            dimension: Name of the dimension, only used in the error message.

        Returns:
            The value of one `unit` in the SI base unit.

        Raises:
            ValueError: If the unit is not in the factor table.
        """
        factor = factor_table.get(unit.lower())
        if factor is None:
            raise ValueError(f'Unknown unit {unit!r} for the dimension {dimension!r}.')
        return factor
