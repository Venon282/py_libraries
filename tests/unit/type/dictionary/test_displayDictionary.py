"""Unit tests for displayDictionary."""

from collections.abc import Mapping
from types import MappingProxyType

import pytest

from py_libraries.type.dictionary import displayDictionary


def countKeys(dct: Mapping[str, object]) -> int:
    """Count the keys of a dictionary, including the keys of nested dicts.

    Args:
        dct: Dictionary to inspect.

    Returns:
        The total number of keys.
    """
    return sum(
        1 + (countKeys(value) if isinstance(value, dict) else 0)
        for value in dct.values()
    )


def test_flat_dictionary_has_one_line_per_key() -> None:
    """Write `key: value` on its own line for each entry."""
    assert displayDictionary({'a': 1, 'b': 'x', 'c': 2.5}) == 'a: 1\nb: x\nc: 2.5'


def test_nested_dictionaries_are_indented() -> None:
    """Indent the keys of a nested dict one level deeper than its parent key."""
    dct = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}, 'f': 4}

    expected = 'a: 1\nb:\n    c: 2\n    d:\n        e: 3\nf: 4'
    assert displayDictionary(dct) == expected


def test_empty_dictionary_gives_an_empty_string() -> None:
    """Return an empty string when there is nothing to display."""
    assert displayDictionary({}) == ''


def test_result_has_no_trailing_newline() -> None:
    """Do not end the string with a line break."""
    assert not displayDictionary({'a': {'b': 1}}).endswith('\n')


@pytest.mark.parametrize(
    ('indent', 'expected'),
    [('--', 'a:\n--b: 1'), ('\t', 'a:\n\tb: 1'), ('', 'a:\nb: 1')],
)
def test_custom_indent(indent: str, expected: str) -> None:
    """Use the given string for one indentation level.

    Args:
        indent: String of one indentation level.
        expected: Expected result.
    """
    assert displayDictionary({'a': {'b': 1}}, indent=indent) == expected


def test_index_sets_the_starting_indentation_level() -> None:
    """Start at a deeper level when `index` is given."""
    result = displayDictionary({'a': {'b': 1}}, indent='  ', index=2)

    assert result == '    a:\n      b: 1'


@pytest.mark.parametrize(
    ('value', 'text'),
    [
        (None, 'None'),
        (True, 'True'),
        ([1, 2], '[1, 2]'),
        ((1, 2), '(1, 2)'),
        ('text', 'text'),
    ],
)
def test_values_are_written_with_str(value: object, text: str) -> None:
    """Write any non-dict value with its string form.

    Args:
        value: Value of the entry.
        text: Expected text after the key.
    """
    assert displayDictionary({'k': value}) == f'k: {text}'


def test_insertion_order_is_kept() -> None:
    """Write the keys in the order of the dictionary, not sorted."""
    assert displayDictionary({'z': 1, 'a': 2}).splitlines() == ['z: 1', 'a: 2']


def test_one_line_per_key_for_non_empty_nested_dictionaries() -> None:
    """Write exactly one line for each key at any depth."""
    dct = {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}, 'f': {'g': 4}, 'h': 5}

    assert len(displayDictionary(dct).splitlines()) == countKeys(dct)


def test_mapping_that_is_not_a_dict_is_accepted_at_the_top_level() -> None:
    """Display a read-only mapping."""
    assert displayDictionary(MappingProxyType({'a': 1})) == 'a: 1'
