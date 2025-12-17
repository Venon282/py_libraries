class Constant(type):
    def __setattr__(cls, key, value):
        raise TypeError(f"Cannot reassign constant '{key}'")