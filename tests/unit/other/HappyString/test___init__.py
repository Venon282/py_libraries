"""Unit tests for HappyString.__init__."""

from py_libraries.other.HappyString import HappyString


def test___init__stores_the_allowed_chars() -> None:
    """Keep the characters that were given."""
    happy = HappyString('ab')

    assert happy.chars == 'ab'


def test___init__defaults_to_abc() -> None:
    """Use `abc` when no characters are given."""
    happy = HappyString()

    assert happy.chars == 'abc'
