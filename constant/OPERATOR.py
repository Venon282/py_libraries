from .Constant import ConstantMeta
import operator as op
import numpy as np

class OPERATOR(metaclass=ConstantMeta):
    OPERATORS_MAP = {
        "==": op.eq,
        "!=": op.ne,
        ">":  op.gt,
        "<":  op.lt,
        "<=": op.le,
        ">=": op.ge,
        "+":  op.add,
        "-":  op.sub,
        "*":  op.mul,
        "/":  op.truediv,
        "//": op.floordiv,
        "%":  op.mod,
        "**": op.pow,
        "in": lambda a, b: op.contains(b, a),   # usage: a in b
    }

    # Logical operations
    LOGIC_MAP = {
        "and": op.and_,
        "or":  op.or_,
        "not": op.not_,
    }

    # String‐to‐boolean helper
    STR2BOOL = {
        "true":  True,  "t":     True,  "yes":  True,  "y":    True,  "1": True,
        "false": False, "f":     False, "no":   False, "n":    False, "0": False,
    }
    
    EMPTY = (None, np.nan, "", (), [], {}, set())