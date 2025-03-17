from sklearn.model_selection import train_test_split

def trainTestSplit(*args, test_size, random_state):
    if test_size == 0.0:
        return tuple(x for obj in args for x in (obj, type(obj)()))
    elif test_size == 1.0:
        return tuple(x for obj in args for x in (type(obj)(), obj))
    else:        
        return train_test_split(*args, test_size=test_size, random_state=random_state)