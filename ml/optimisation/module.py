import sys

def setHyperparameter(method, fixe_hparams, name, sub_names=[], **kwargs): 
    # If name in fix parameters   
    if name in fixe_hparams:
        hp = method.__self__
        return hp.Fixed(name, fixe_hparams[name], parent_name=kwargs.pop('parent_name', None), parent_values=kwargs.pop('parent_values', None))

    # Verify if another possibilities is in subnames
    if isinstance(sub_names, list):
        for sub_name in sub_names:
            # If a valid subname, processs
            if sub_name in fixe_hparams:
                sub_names=sub_name
                break
        else: sub_names = ''
    
    # If a valid subnames, process it
    if sub_names and sub_names in fixe_hparams:
        hp = method.__self__
        return hp.Fixed(name, fixe_hparams[sub_names], parent_name=kwargs.pop('parent_name', None), parent_values=kwargs.pop('parent_values', None))

    return method(name, **kwargs)