import numpy as np

REGEX_NUMBER = r'[-+]?\d+(?:[,\.]\d+)?'
REGEX_SCIENTIFIC_NUMBER = REGEX_NUMBER + r'e[+-]?\d+'

SPACE = '               '
EMPTY = [None, np.nan, ""] 