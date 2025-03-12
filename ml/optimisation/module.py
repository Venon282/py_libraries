import sys

def setHyperparameter(method, fixe_hparams, name, **kwargs):
    if name in fixe_hparams:
        hp = method.__self__
        return hp.Fixed(name, fixe_hparams[name], parent_name=kwargs.pop('parent_name', None), parent_values=kwargs.pop('parent_values', None))
    else:
        return method(name, **kwargs)