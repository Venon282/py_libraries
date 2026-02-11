class ConstantMeta(type):
    def __setattr__(cls, key, value):
        if key in cls.__dict__:
            raise TypeError(f"Cannot reassign constant '{key}'")
        super().__setattr__(key, value)
    
class Constants(metaclass=ConstantMeta):
    ...