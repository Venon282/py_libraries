from .Constant import Constant

class EXTENSION(metaclass=Constant):
    IMAGE    = ("png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif", "webp")
    TEXT     = ("txt",)
    CODE     = ("py", "js", "java", "cpp", "c", "cs", "rb", "go", "ts", "php", "rs", "swift", "kt")
    DOCUMENT = ("pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "odt", "ods")
    ARCHIVE  = ("zip", "tar", "gz", "bz2", "xz", "rar", "7z")
    DATA     = ("csv", "json", "xml", "yaml", "yml", "parquet")