import numpy as np
import operator as op
import re

class ConstMeta(type):
    def __setattr__(cls, key, value):
        raise TypeError(f"Cannot reassign constant '{key}'")

class CONST(metaclass=ConstMeta):
    """Immutable container of constants."""
    
    # -------------------------------------------------------------------------
    # Regex patterns
    # -------------------------------------------------------------------------
    class REGEX(metaclass=ConstMeta):
        # Numbers
        INTEGER            = r'[+-]?\d+'
        FLOAT              = r'[+-]?\d+(?:[.,]\d+)?'
        SCIENTIFIC_NUMBER  = FLOAT + r'[eE][+-]?\d+'
        HEX_NUMBER         = r'0[xX][0-9a-fA-F]+'
        BINARY_NUMBER      = r'0[bB][01]+'
        OCTAL_NUMBER       = r'0[oO][0-7]+'

        # Words, identifiers
        WORD               = r'\w+'
        VARIABLE_NAME      = r'[A-Za-z_]\w*'
        WHITESPACE         = r'\s+'
        NON_WHITESPACE     = r'\S+'

        # Common data types
        EMAIL              = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
        URL                = (
            r'https?://'                                           # protocol
            r'(?:[A-Za-z0-9\.-]+)'                                 # domain
            r'(?:\:\d+)?'                                          # optional port
            r'(?:/[A-Za-z0-9\._\-/]*)*'                            # path
            r'(?:\?[A-Za-z0-9=&;%\-_]*)?'                          # query
        )
        IP_V4              = r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3}'
        UUID               = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}'
        ISO_DATE           = r'\d{4}-\d{2}-\d{2}'
        ISO_TIME_24        = r'(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?'
        ISO_DATETIME       = ISO_DATE + r'[T ]' + ISO_TIME_24

        # Python‐specific
        class PYTHON(metaclass=ConstMeta):
            COMMENT_BLOCK    = r'(?s)(?:\'\'\'[\s\S]*?\'\'\'|\"\"\"[\s\S]*?\"\"\")'
            SINGLE_COMMENT   = r'#.*'
            STRING_LITERAL   = r'(?:r?["\'])(?:\\.|[^\\\n])*?["\']'

    # -------------------------------------------------------------------------
    # File extensions
    # -------------------------------------------------------------------------
    class EXTENSION(metaclass=ConstMeta):
        IMAGE    = ("png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif", "webp")
        TEXT     = ("txt",)
        CODE     = ("py", "js", "java", "cpp", "c", "cs", "rb", "go", "ts", "php", "rs", "swift", "kt")
        DOCUMENT = ("pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "odt", "ods")
        ARCHIVE  = ("zip", "tar", "gz", "bz2", "xz", "rar", "7z")
        DATA     = ("csv", "json", "xml", "yaml", "yml", "parquet")
    
    # -------------------------------------------------------------------------
    # “Empty” or missing values
    # -------------------------------------------------------------------------
    EMPTY = (None, np.nan, "", (), [], {}, set())

    # -------------------------------------------------------------------------
    # Operator mappings
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Miscellaneous
    # -------------------------------------------------------------------------
    VERSION = "1.0.0"
    PI      = np.pi
    E       = np.e

    DAYS_OF_WEEK = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    MONTHS       = (
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    )

# Example usage (will raise if you try to overwrite):
# CONST.PI = 3.14            # TypeError
# CONST.REGEX.URL = "foo"    # TypeError
