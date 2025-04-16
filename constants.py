import numpy as np
import operator as op

class ConstMeta(type):
    def __setattr__(cls, key, value):
        raise TypeError("Les attributs de constantes ne peuvent pas être modifiés")

class CONST(metaclass=ConstMeta):
    class REGEX(metaclass=ConstMeta):
        NUMBER = r'[-+]?\d+(?:[,\.]\d+)?'
        SCIENTIFIC_NUMBER = NUMBER + r'[eE][+-]?\d+'
        
        class PYTHON(metaclass=ConstMeta):
            COMMENT_BLOCK = r'(?s)(?:\'\'\'[\s\S]*?\'\'\'|\"\"\"[\s\S]*?\"\"\")'
    
    EMPTY = (None, np.nan, "")
    OPERATORS_MAP = {
        "==": op.eq,
        "!=": op.ne,
        ">": op.gt,
        "<": op.lt,
        "<=": op.le,
        ">=": op.ge,
    }



