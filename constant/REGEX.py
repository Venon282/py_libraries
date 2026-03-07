from.Constant import ConstantMeta

class REGEX(metaclass=ConstantMeta):
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