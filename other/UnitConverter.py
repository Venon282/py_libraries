from __future__ import annotations

_LENGTH_FACTORS: dict[str, float] = {
    # SI base: metre
    "metre":       1.0,
    "kilometer":   1_000.0,
    "centimetre":  1e-2,
    "millimetre":  1e-3,
    "micrometre":  1e-6,
    "nanometre":   1e-9,
    "angstrom":    1e-10,
    "inch":        0.0254,
    "foot":        0.3048,
    "yard":        0.9144,
    "mile":        1_609.344,
    "nautical_mile": 1_852.0,
    "light_year":  9.460_730_472_580_8e15,
}

_MASS_FACTORS: dict[str, float] = {
    # SI base: kilogram
    "kilogram":    1.0,
    "gram":        1e-3,
    "milligram":   1e-6,
    "microgram":   1e-9,
    "tonne":       1_000.0,
    "pound":       0.453_592_37,
    "ounce":       0.028_349_523_125,
    "stone":       6.350_293_18,
    "short_ton":   907.184_74,
    "long_ton":    1_016.046_9088,
}

_TIME_FACTORS: dict[str, float] = {
    # SI base: second
    "second":      1.0,
    "millisecond": 1e-3,
    "microsecond": 1e-6,
    "nanosecond":  1e-9,
    "minute":      60.0,
    "hour":        3_600.0,
    "day":         86_400.0,
    "week":        604_800.0,
    "year":        31_557_600.0,  # Julian year
}

_TEMPERATURE_UNITS = {"celsius", "fahrenheit", "kelvin", "rankine"}

_AREA_FACTORS: dict[str, float] = {
    # SI base: square metre
    "square_metre":      1.0,
    "square_kilometre":  1e6,
    "square_centimetre": 1e-4,
    "square_millimetre": 1e-6,
    "square_inch":       6.4516e-4,
    "square_foot":       0.092_903_04,
    "square_yard":       0.836_127_36,
    "square_mile":       2_589_988.110_336,
    "acre":              4_046.856_422_4,
    "hectare":           10_000.0,
}

_INVERSE_AREA_FACTORS: dict[str, float] = {
    # SI base: per square metre (m^-2)
    # Factor = how many m^-2 equals 1 of this unit
    "per_square_metre":       1.0,
    "per_square_kilometre":   1e-6,
    "per_square_centimetre":  1e4,
    "per_square_millimetre":  1e6,
    "per_square_micrometre":  1e12,
    "per_square_nanometre":   1e18,
    "per_square_angstrom":    1e20,
    "per_square_inch":        1.0 / 6.4516e-4,
    "per_square_foot":        1.0 / 0.092_903_04,
    "per_square_yard":        1.0 / 0.836_127_36,
    "per_square_mile":        1.0 / 2_589_988.110_336,
}

_VOLUME_FACTORS: dict[str, float] = {
    # SI base: cubic metre
    "cubic_metre":       1.0,
    "litre":             1e-3,
    "millilitre":        1e-6,
    "centilitre":        1e-5,
    "cubic_centimetre":  1e-6,
    "angstrom_cubed":    1e-30,
    "cubic_inch":        1.638_706_4e-5,
    "cubic_foot":        0.028_316_846_592,
    "cubic_yard":        0.764_554_857_984,
    "gallon_us":         3.785_411_784e-3,
    "gallon_uk":         4.546_09e-3,
    "quart_us":          9.463_529_46e-4,
    "pint_us":           4.731_764_73e-4,
    "pint_uk":           5.682_612_5e-4,
    "fluid_ounce_us":    2.957_352_956_25e-5,
    "fluid_ounce_uk":    2.841_306_25e-5,
    "tablespoon":        1.478_676_478_125e-5,
    "teaspoon":          4.928_921_593_75e-6,
}

_SPEED_FACTORS: dict[str, float] = {
    # SI base: metre per second
    "metre_per_second":    1.0,
    "kilometre_per_hour":  1.0 / 3.6,
    "mile_per_hour":       0.447_04,
    "foot_per_second":     0.3048,
    "knot":                0.514_444,
}

_PRESSURE_FACTORS: dict[str, float] = {
    # SI base: pascal
    "pascal":        1.0,
    "kilopascal":    1_000.0,
    "megapascal":    1_000_000.0,
    "bar":           100_000.0,
    "millibar":      100.0,
    "atmosphere":    101_325.0,
    "torr":          133.322_368_421,
    "mmhg":          133.322_387_415,
    "psi":           6_894.757_293_168,
}

_ENERGY_FACTORS: dict[str, float] = {
    # SI base: joule
    "joule":            1.0,
    "kilojoule":        1_000.0,
    "megajoule":        1_000_000.0,
    "calorie":          4.184,
    "kilocalorie":      4_184.0,
    "watt_hour":        3_600.0,
    "kilowatt_hour":    3_600_000.0,
    "electron_volt":    1.602_176_634e-19,
    "btu":              1_055.056,
    "foot_pound":       1.355_817_948_331_4,
    "erg":              1e-7,
}

_POWER_FACTORS: dict[str, float] = {
    # SI base: watt
    "watt":            1.0,
    "kilowatt":        1_000.0,
    "megawatt":        1_000_000.0,
    "milliwatt":       1e-3,
    "horsepower":      745.699_871_582_270_22,
    "btu_per_hour":    0.293_071,
    "foot_pound_per_second": 1.355_817_948_331_4,
}

_DATA_FACTORS: dict[str, float] = {
    # SI base: bit
    "bit":       1.0,
    "nibble":    4.0,
    "byte":      8.0,
    "kilobit":   1_000.0,
    "kilobyte":  8_000.0,
    "megabit":   1_000_000.0,
    "megabyte":  8_000_000.0,
    "gigabit":   1e9,
    "gigabyte":  8e9,
    "terabit":   1e12,
    "terabyte":  8e12,
    "petabit":   1e15,
    "petabyte":  8e15,
    "kibibyte":  8.0 * 1_024,
    "mebibyte":  8.0 * 1_024 ** 2,
    "gibibyte":  8.0 * 1_024 ** 3,
    "tebibyte":  8.0 * 1_024 ** 4,
}

_ANGLE_FACTORS: dict[str, float] = {
    # SI base: radian
    "radian":     1.0,
    "degree":     3.141_592_653_589_793 / 180.0,
    "gradian":    3.141_592_653_589_793 / 200.0,
    "arcminute":  3.141_592_653_589_793 / 10_800.0,
    "arcsecond":  3.141_592_653_589_793 / 648_000.0,
    "turn":       2.0 * 3.141_592_653_589_793,
}

# Map each dimension name to its factor table.
_DIMENSION_MAP: dict[str, dict[str, float]] = {
    "length":      _LENGTH_FACTORS,
    "mass":        _MASS_FACTORS,
    "time":        _TIME_FACTORS,
    "area":        _AREA_FACTORS,
    "inverse_area":_INVERSE_AREA_FACTORS,
    "volume":      _VOLUME_FACTORS,
    "speed":       _SPEED_FACTORS,
    "pressure":    _PRESSURE_FACTORS,
    "energy":      _ENERGY_FACTORS,
    "power":       _POWER_FACTORS,
    "data":        _DATA_FACTORS,
    "angle":       _ANGLE_FACTORS,
}


# ---------------------------------------------------------------------------
# Temperature helpers (offset conversions cannot use simple multipliers)
# ---------------------------------------------------------------------------

def _toKelvin(value: float, unit: str) -> float:
    """Convert a temperature value to Kelvin."""
    unit = unit.lower()
    if unit == "kelvin":
        return value
    if unit == "celsius":
        return value + 273.15
    if unit == "fahrenheit":
        return (value + 459.67) * 5.0 / 9.0
    if unit == "rankine":
        return value * 5.0 / 9.0
    raise ValueError(f"Unknown temperature unit: '{unit}'")


def _fromKelvin(value_k: float, unit: str) -> float:
    """Convert a Kelvin value to the target temperature unit."""
    unit = unit.lower()
    if unit == "kelvin":
        return value_k
    if unit == "celsius":
        return value_k - 273.15
    if unit == "fahrenheit":
        return value_k * 9.0 / 5.0 - 459.67
    if unit == "rankine":
        return value_k * 9.0 / 5.0
    raise ValueError(f"Unknown temperature unit: '{unit}'")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class UnitConverter:
    """
    Convert numeric values between units within a given physical dimension.

    Supported dimensions
    --------------------
    length, mass, time, temperature, area, volume, speed,
    pressure, energy, power, data, angle

    Usage
    -----
    >>> converter = UnitConverter()
    >>> converter.convert(100, "kilometre", "mile", "length")
    62.13711922369478
    >>> converter.convert(100, "celsius", "fahrenheit", "temperature")
    212.0
    >>> converter.listUnits("pressure")
    ['atmosphere', 'bar', 'kilopascal', ...]
    """

    def convert(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        dimension: str,
    ) -> float:
        """
        Convert *value* from *from_unit* to *to_unit* within *dimension*.

        Parameters
        ----------
        value      : numeric value to convert
        from_unit  : source unit name (case-insensitive)
        to_unit    : target unit name (case-insensitive)
        dimension  : physical dimension (case-insensitive)

        Returns
        -------
        float : converted value

        Raises
        ------
        ValueError : unknown dimension, from_unit, or to_unit
        """
        dim_key = dimension.lower()
        from_key = from_unit.lower()
        to_key = to_unit.lower()

        if dim_key == "temperature":
            return self._convertTemperature(value, from_key, to_key)

        factor_table = self._getFactorTable(dim_key)
        from_factor = self._getFactor(factor_table, from_key, from_unit)
        to_factor = self._getFactor(factor_table, to_key, to_unit)

        return value * from_factor / to_factor

    def listDimensions(self) -> list[str]:
        """Return all supported dimension names, sorted alphabetically."""
        all_dimensions = list(_DIMENSION_MAP.keys()) + ["temperature"]
        return sorted(all_dimensions)

    def listUnits(self, dimension: str) -> list[str]:
        """
        Return all unit names available for *dimension*, sorted alphabetically.

        Raises
        ------
        ValueError : unknown dimension
        """
        dim_key = dimension.lower()
        if dim_key == "temperature":
            return sorted(_TEMPERATURE_UNITS)

        factor_table = self._getFactorTable(dim_key)
        return sorted(factor_table.keys())

    def isSupported(self, unit: str, dimension: str) -> bool:
        """Return True if *unit* is supported within *dimension*."""
        try:
            return unit.lower() in self.listUnits(dimension)
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _convertTemperature(
        self, value: float, from_unit: str, to_unit: str
    ) -> float:
        """Route temperature conversions through Kelvin as the canonical base."""
        if from_unit not in _TEMPERATURE_UNITS:
            raise ValueError(f"Unknown temperature unit: '{from_unit}'")
        if to_unit not in _TEMPERATURE_UNITS:
            raise ValueError(f"Unknown temperature unit: '{to_unit}'")

        value_k = _toKelvin(value, from_unit)
        return _fromKelvin(value_k, to_unit)

    def _getFactorTable(self, dim_key: str) -> dict[str, float]:
        """Return the factor table for *dim_key* or raise ValueError."""
        factor_table = _DIMENSION_MAP.get(dim_key)
        if factor_table is None:
            supported = ", ".join(self.listDimensions())
            raise ValueError(
                f"Unknown dimension: '{dim_key}'. Supported: {supported}"
            )
        return factor_table

    @staticmethod
    def _getFactor(
        factor_table: dict[str, float], key: str, original: str
    ) -> float:
        """Return the SI factor for *key* or raise ValueError."""
        factor = factor_table.get(key)
        if factor is None:
            raise ValueError(f"Unknown unit: '{original}'")
        return factor
