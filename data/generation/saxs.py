import numpy as np
import h5py
from itertools import product, repeat
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging
import os
import concurrent.futures
from line_profiler import profile

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VolumeShape:
    @staticmethod
    def sphere(radius):
        return np.pi * radius**3 *4/3
    
    @staticmethod
    def parallelepiped(length_a, length_b, length_c):
        return length_a * length_b * length_c
    
    @staticmethod
    def cylinder(radius, length):
        return np.pi * radius**2 * length

class UnitConvertor:
    @staticmethod
    def nmToÅ(data:int|float|np.ndarray):
        return data * 10.0
    
    @staticmethod
    def ÅToNm(data:int|float|np.ndarray):
        return data / 10.0

    @staticmethod
    def cm2ToÅ2(data:int|float|np.ndarray):
        return data * 1e-16
    
    @staticmethod
    def Å3ToCm3(data:int|float|np.ndarray):
        return data * 1e-24


def scatteringLengthDensityCm2(name):
    """In Å^-2"""
    name = name.strip().lower()
    
    values = {
        'au': 1.31596e+12,
        'ag': 7.76211e11,
        'latex': 8.81075e+10, #(not sure ??)
        'sio2': 1.86206e+11,
        'h2o': 9.39845e10,
        'water': 9.39845e10
    }
    
    if name in values:
        return values[name]
    
    raise ValueError(f"Unknow {name} material")

Model = None
@profile
def _globalInitModel(q, shape):
    global Model
    from sasmodels.core import load_model
    from sasmodels.direct_model import DirectModel
    from sasmodels.data import empty_data1D
    
    empty_data_1d = empty_data1D(q)
    model_def = load_model(shape)
    Model = DirectModel(empty_data_1d, model_def)

@profile
def generateSignal(params: dict, shape: str, material_sld_val, solvent_sld_val):
    """
    Worker executed in each process. 
    `params` should include 'concentration' and the geometric parameters (radius, length_a, ...).
    Returns a tuple (params_dict, intensity_array).
    """
    global Model

    # extract concentration and geometric params
    params = dict(params)  # copy
    concentration = float(params.pop('concentration'))

    # compute volume in Å^3 using the requested shape function
    if not hasattr(VolumeShape, shape):
        raise ValueError(f"Unknown shape '{shape}' for volume computation.")
    
    volume_A3 = getattr(VolumeShape, shape)(**params)
    volume_cm3 = UnitConvertor.Å3ToCm3(volume_A3)

    scale = concentration * volume_cm3

    # prepare model parameters (merge dims back)
    model_pars = dict(scale=scale, background=0.0, sld=material_sld_val, sld_solvent=solvent_sld_val)
    model_pars.update(params)

    intensity = Model(**model_pars) * 1e12
    
    for key, value in params.items():
        params[key] = UnitConvertor.ÅToNm(value)
    params['concentration'] = concentration    
    model_pars.update(params)

    return {'params': model_pars, 'intensity': intensity}
    
@profile
def main(
    q:list,
    parameters:dict[list],
    shape:str,
    material:str,
    env:str,
    other_attrs:dict={},
    parameters_operator:str='product',
    max_workers:int=None,
    save_h5_filepath='./signals.h5',
):
    save_h5_filepath = safePath(save_h5_filepath)
    material_sld = UnitConvertor.cm2ToÅ2(scatteringLengthDensityCm2(material))
    solvent_sld = UnitConvertor.cm2ToÅ2(scatteringLengthDensityCm2(env))
    
    logger.debug([[k,len(v)] for k, v in parameters.items()])
    if parameters_operator == 'product':
        iterator = [
            dict(zip(parameters.keys(), values))
            for values in product(*parameters.values())
        ]
        n_signal = np.prod([len(v) for v in parameters.values()]).astype(int)
    elif parameters_operator == 'stack':
        # All list must have the same size
        lengths = [len(v) for v in parameters.values()]
        if len(set(lengths)) != 1:
            raise ValueError("All lists must have the same length for 'stack' operator.")

        # Generate dictionnary list
        iterator = [dict(zip(parameters.keys(), values)) for values in zip(*parameters.values())]
        n_signal = lengths[0]
    else: raise ValueError(f'parameters_operator must be either product or stack.')
    
    logger.info(f'{n_signal} signal of {material} {shape} will be generate.')
    
    # Open HDF5 file
    with h5py.File(save_h5_filepath, 'w') as f:
        logger.info(f'Dataset creations...')
        # Create datasets with chunking & compression
        dset_intensities = f.create_dataset(
            'intensities',
            shape=(n_signal, len(q)),
            dtype=np.float32,
            chunks=True,
        )
        dset_q = f.create_dataset('q', data=q)
        dsets_meta = {}
        for k in parameters.keys():
            dsets_meta[k] = f.create_dataset(k, shape=(n_signal,), dtype=np.float64, chunks=True)

        logger.info(f'Starting the generation with {os.cpu_count()} CPU...')
        if max_workers == 0:
            _globalInitModel(q, shape)  
            for i, params in enumerate(tqdm(iterator, total=n_signal, desc="Generating signals", mininterval=1, miniters=100)):
                res = generateSignal(params, shape, material_sld, solvent_sld)
                dset_intensities[i, :] = res['intensity']
                for k in parameters.keys():
                    dsets_meta[k][i] = res['params'][k]
        else:
            # Parallel mode
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_globalInitModel,
                initargs=(q, shape)
            ) as executor:
                futures = executor.map(
                    generateSignal,
                    iterator,
                    repeat(shape),
                    repeat(material_sld),
                    repeat(solvent_sld)
                )
                for i, res in enumerate(tqdm(futures, total=n_signal, desc="Generating signals", mininterval=1, miniters=100)):
                    dset_intensities[i, :] = res['intensity']
                    for k in parameters.keys():
                        dsets_meta[k][i] = res['params'][k]
        f.attrs["q"] = q
        f.attrs["material"] = material
        f.attrs["technique"] = "saxs"
        f.attrs["techno"] = "sasmodels"
        f.attrs["shape"] = shape
        f.attrs["environment"] = env
        for key, value in other_attrs.items():
            f.attrs[key] = value
        
    logger.info(f"All signals written to {save_h5_filepath}.")
    
def safePath(path, suffix="_", max_tries=1000):
    """Create a unique name next to the original.
    Returns the path of the created copy.
    """
    if not os.path.exists(path):
        return path
    
    folder, filename = os.path.split(path)
    base, ext = os.path.splitext(filename)
    
    for i in range(max_tries):
        new_name = f"{base}{suffix}{i}{ext}"
            
        new_path = os.path.join(folder, new_name)
        if not os.path.exists(new_path):
            return new_path
        
    raise FileExistsError(f"Could not create a unique path of {path} after {max_tries} attempts")

if __name__ == '__main__':
    # py -m kernprof -l -v saxs_simulation_generation.py > report.txt
    # -l line by line -v in the consol > consol display in a txt
    q = np.linspace(1e-3, 1, 1000)
    concentrations = np.logspace(8, 19, 1000).astype(np.float64)
    shape = "cube"
    material = "ag"
    env = "water"
    parameters_operator = 'product'
    save_h5_filepath = rf'C:\Users\ET281306\Downloads\{material}_{shape}.h5'
    max_workers=0
    other_attrs = {
        "author":"Esteban THEVENON"
    }
    
    # define parameters
    if shape == 'cube':
        parameters_operator = 'stack'
        shape = 'parallelepiped'
        other_attrs['shape'] = 'cube'
        length=np.arange(10, 101, 1)
        final_length = np.repeat(length, len(concentrations)) # [1, 2, 3] -> [1, 1, 2, 2, 3, 3,]
        parameters = {
            'concentration':np.array(list(concentrations) * len(length)), # [1, 2, 3] -> [1, 2, 3, 1, 2, 3]
            'length_a': UnitConvertor.nmToÅ(final_length),   # height
            'length_b': UnitConvertor.nmToÅ(final_length), # width
            'length_c': UnitConvertor.nmToÅ(final_length)  # length
            }
    elif shape == 'parallelepiped':
        parameters = {
            'concentration':concentrations,
            'length_a': UnitConvertor.nmToÅ(np.arange(10, 51, 5)),   # height
            'length_b': UnitConvertor.nmToÅ(np.arange(20, 101, 10)), # width
            'length_c': UnitConvertor.nmToÅ(np.arange(50, 201, 10))  # length
            }
    elif shape == 'sphere':
        parameters = {
        'concentration':concentrations,
            'radius': UnitConvertor.nmToÅ(np.arange(10, 101, 20)/2)
            }
    elif shape == 'cylinder':
        parameters = {
        'concentration':concentrations,
            'radius': UnitConvertor.nmToÅ(np.arange(10, 101, 20)/2),
            'length': UnitConvertor.nmToÅ(np.arange(5, 16, 4)),
            }
    else:
        raise Exception(f'{shape} parameters is not define. Please set this shape before going farwer.')
    
    main(
        q = q,
        parameters = parameters,
        parameters_operator = parameters_operator,
        shape = shape,
        material = material,
        env=env,
        other_attrs=other_attrs,
        save_h5_filepath=save_h5_filepath,
        max_workers=max_workers,
    )