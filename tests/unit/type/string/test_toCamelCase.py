import pytest
from py_libraries.type.string import toCamelCase


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('hello world foo', 'helloWorldFoo'),
        ('Hello WORLD', 'helloWorld'),
        ('ABC', 'abc'),
        ('', ''),
        ('single', 'single'),
    ],
)
def test_toCamelCase_default_separator(s: str, expected: str) -> None:
    assert toCamelCase(s) == expected


def test_toCamelCase_custom_separator() -> None:
    assert toCamelCase('my_var_name', separator='_') == 'myVarName'


def test_toCamelCase_separator_absent_from_the_string_only_lowercases_it() -> None:
    assert toCamelCase('Hello World', separator='_') == 'hello world'
