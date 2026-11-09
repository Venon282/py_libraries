from py_libraries.other.HappyString import HappyString


def test___init__stores_the_allowed_chars() -> None:
    happy = HappyString('ab')

    assert happy.chars == 'ab'


def test___init__defaults_to_abc() -> None:
    happy = HappyString()

    assert happy.chars == 'abc'
