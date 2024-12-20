import pickle
import warnings
import statistics
import joblib

def proportions(lst):
    return {str(element): list(lst).count(element) / len(lst) for element in set(lst)}

def countElements(lst):
    return {str(element): list(lst).count(element) for element in set(lst)}

def save(lst, path='./lst.pkl'):
    return joblib.dump(lst, path)
        
def load(path='./lst.pickle'):
    return joblib.load(path)
    
def interpretor(x_targ, x, y):
    def Jup(j, x):
        return j if j+1>=len(x) else j+1
    
    new_y = []
    j = 0
    for xt in x_targ:
        if xt < x[j]:
            if j == 0:
                new_y.append(y[j]) # If our y start later than the need we have, keep the first value of val # todo improve by checking the curve direction
            else:
                distance_max = x[j] - x[j-1]
                distance = x[j] - xt
                ratio = distance / distance_max
                new_y.append(y[j] - (y[j] - y[j-1]) * ratio)
                j=Jup(j, x)
        elif xt > x[j]:
            while j+1 < len(x) and xt > x[j+1]: # catch up the late if j have more than 1 value bellow the need wavelength
                j=Jup(j, x)
            if j+1 >= len(x):
                new_y.append(y[j]) # not anymore j value # todo improve by checking the curve direction
            else:
                distance_max = x[j+1] - x[j]
                distance = xt - x[j]
                ratio = distance / distance_max
                warnings.filterwarnings('error')
                try:
                    new_y.append(y[j] + (y[j+1] - y[j]) * ratio)
                except Warning:
                    print(j)
                    print(y[j])
                    print(y[j+1])
                    print(ratio)
                    raise
                j=Jup(j, x)
        else: # if the wavelength are equals
            new_y.append(y[j])
            j=Jup(j, x)
    return new_y

def smoothMiddle(lst, window=5):
    """Smooth with the current element place on the middle of the window

    Args:
        lst (_type_): _description_
        window (int, optional): _description_. Defaults to 5. Min is 1.
    """
    new_lst = []
    shift = max(1, window//2)
    shiftl, shiftr = (shift, shift) if window%2==1 else (shift-1, shift)
    return [statistics.mean(lst[max(0, i-shiftl):min(len(lst), i+shiftr+1)]) for i, l in enumerate(lst)]
        
    