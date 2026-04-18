import numpy as np
import h5py
from itertools import product, repeat
from tqdm import tqdm
import os
import concurrent.futures
from line_profiler import profile

# Small Angular X-ray Scatering

#? doc for add more shape: https://www.sasview.org/docs/user/qtgui/Perspectives/Fitting/models/index.html
from ...other.loggingUtils import getLogger
from ...constant.Constant import Constants

logger = getLogger(__name__)

"""
Scattering Length Density in cm^2

https://slddb.reflectometry.org/

        1. look for the desire one
        2. click on select
        3. Choose Cu-Ka or Mo-Ka
        4. Choose SLD unit
        5. Multiply it by the defined value (eg: (10⁻⁶ Å⁻²) -> multiply by *10**(-6))
        6. Use the UnitConvertor.Å2ToCm2
"""
Constants.SLD = {
    'h2o_21c': 9.44948e+10,
    'au': 1.25387e+12,
    'ag': 7.82265e+11,
    'fe': 5.98198e+11,
    'sio2': 1.88952e+11,
    'h8c8_latex': 9.60732e+10,
}
Constants.__annotations__['SLD'] = dict[str, float]

"""
To ensure that the parameters pass are valid depending of the shape
"""
Constants.SHAPE_PARAM = {
    'sphere': {'radius'},
    'cylinder': {'radius', 'length'},
    'parallelepiped': {'length_a', 'length_b', 'length_c'},
}
Constants.__annotations__['SHAPE_PARAM'] = dict[str, set[str]]

def _validateParameters(shape: str, params: dict[str, set[str]]) -> None:
    expected = Constants.SHAPE_PARAM.get(shape)

    if expected is None:
        raise ValueError(f"Unknown shape '{shape}'.")

    geometric_keys = set(params) - {'concentration'}
    missing = expected - geometric_keys
    extra = geometric_keys - expected

    if missing:
        raise ValueError(f"Missing parameters for '{shape}': {missing}")

    if extra:
        raise ValueError(f"Unexpected parameters for '{shape}': {extra}")

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
    """
    A utility class for converting between metric and Angstrom-based units
    commonly used in scattering science.
    """

    @staticmethod
    def nmToÅ(data: int | float | np.ndarray):
        """Converts nanometers to Angstroms (1 nm = 10 Å)."""
        return data * 10.0

    @staticmethod
    def ÅToNm(data: int | float | np.ndarray):
        """Converts Angstroms to nanometers (1 Å = 0.1 nm)."""
        return data / 10.0

    @staticmethod
    def cm2ToÅ2(data: int | float | np.ndarray):
        """
        Converts SLD from cm^-2 to Å^-2.
        Factor: 1 cm^-2 = (10^8 Å)^-2 = 10^-16 Å^-2.
        """
        return data * 1e-16

    @staticmethod
    def Å2ToCm2(data: int | float | np.ndarray):
        """
        Converts SLD from Å^-2 to cm^-2.
        Factor: 1 Å^-2 = (10^-8 cm)^-2 = 10^16 cm^-2.
        """
        return data * 1e16

    @staticmethod
    def Å3ToCm3(data: int | float | np.ndarray):
        """
        Converts volume from Å^3 to cm^3.
        Factor: 1 Å^3 = (10^-8 cm)^3 = 10^-24 cm^3.
        """
        return data * 1e-24

    @staticmethod
    def cm3ToÅ3(data: int | float | np.ndarray):
        """
        Converts volume from cm^3 to Å^3.
        Factor: 1 cm^3 = (10^8 Å)^3 = 10^24 Å^3.
        """
        return data * 1e24


_model_cache: dict[tuple, object] = {}
def _get_model(q: np.ndarray, shape: str):
    key = (tuple(q), shape)
    if key not in _model_cache:
        from sasmodels.core import load_model
        from sasmodels.direct_model import DirectModel
        from sasmodels.data import empty_data1D
        _model_cache[key] = DirectModel(empty_data1D(q), load_model(shape))
    return _model_cache[key]

@dataclass
class SignalResult:
    """
    Holds the output of a single :func:`generateSignal` call.

    Attributes
    ----------
    intensity:
        Scattering intensity array (same length as q).
    params_nm:
        Geometric parameters expressed in **nanometres** (human-readable).
    params_model:
        Raw parameters forwarded to sasmodels (geometric values in **Å**,
        plus scale and background).
    concentration:
        Particle number concentration (cm⁻³).
    """
    intensity: np.ndarray
    params_nm: dict[str, float]
    params_model: dict[str, float]
    concentration: float

@profile
def generateSignal(
    params: dict,
    q:np.ndarray,
    shape: str,
    material_sld_val,
    solvent_sld_val
    ) -> SignalResult:
    """
    Compute a single SAXS intensity curve.

    Parameters
    ----------
    params:
        Must contain ``'concentration'`` and all geometric keys for *shape*,
        with geometric values expressed in **Å**.
    shape:
        sasmodels shape name (e.g. ``'sphere'``, ``'cylinder'``).
    material_sld_val:
        Material SLD in Å⁻².
    solvent_sld_val:
        Solvent SLD in Å⁻².

    Returns
    -------
    SignalResult
    """
    # compute volume in Å^3 using the requested shape function
    if not hasattr(VolumeShape, shape):
        raise ValueError(
            f"VolumeShape has no method for '{shape}'. "
            f"Available: {[m for m in dir(VolumeShape) if not m.startswith('_')]}"
        )

    Model = _get_model(q, shape)

    # extract concentration and geometric params
    params = dict(params)  # copy
    concentration = float(params.pop('concentration'))

    volume_A3 = getattr(VolumeShape, shape)(**params)
    volume_cm3 = UnitConvertor.Å3ToCm3(volume_A3)
    scale = concentration * volume_cm3

    # prepare model parameters (merge dims back)
    params_model: dict[str, float] = {
        "scale":       scale,
        "background":  0.0,
        "sld":         material_sld_val,
        "sld_solvent": solvent_sld_val,
        **params,                         # geometric params in Å
    }

    intensity = Model(**params_model) * 1e12

    params_nm = {k: UnitConvertor.ÅToNm(v) for k, v in params.items()}

    return SignalResult(
        intensity=intensity,
        params_nm=params_nm,
        params_model=params_model,
        concentration=concentration,
    )


def buildParameterGrid(parameters: dict, operator: str) -> list[dict]:
    """
    Expand *parameters* into a flat list of per-signal parameter dicts.

    Parameters
    ----------
    parameters:
        Mapping of parameter name → list of values.
    operator:
        ``'product'`` - cartesian product of all lists.

        ``'stack'``   - element-wise zip (all lists must have equal length).

    Returns
    -------
    list of dict
        Each dict maps parameter names to a single numeric value.
    """
    if operator == 'product':
        return [
            dict(zip(parameters.keys(), values))
            for values in product(*parameters.values())
        ]

    elif operator == 'stack':
        # All list must have the same size
        lengths = [len(v) for v in parameters.values()]
        if len(set(lengths)) != 1:
            raise ValueError(
                "All parameter lists must have equal length for operator='stack'. "
                f"Got lengths: { {k: len(v) for k, v in parameters.items()} }"
            )

        # Generate dictionnary list
        return [
            dict(zip(parameters.keys(), values))
            for values in zip(*parameters.values())
        ]
    else: raise ValueError(f'parameters_operator must be either product or stack.')



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
    _validateParameters(shape, parameters)

    # ensure materials consistency
    material = material.strip().lower()
    env = env.strip().lower()

    save_h5_filepath = safePath(save_h5_filepath)
    material_sld_cm2 = Constants.SLD[material]
    material_sld = UnitConvertor.cm2ToÅ2(material_sld_cm2)
    solvent_sld_cm2 = Constants.SLD[env]
    solvent_sld = UnitConvertor.cm2ToÅ2(solvent_sld_cm2)

    logger.debug("Parameter sizes: %s", {k: len(v) for k, v in parameters.items()})

    param_grid = buildParameterGrid(parameters, parameters_operator)
    n_signal = len(param_grid)

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
        #dset_q = f.create_dataset('q', data=q)
        dsets_meta = {}
        for k in parameters.keys():
            dsets_meta[k] = f.create_dataset(k, shape=(n_signal,), dtype=np.float64, chunks=True)

        def _writeH5(res:SignalResult, index:int):
            dset_intensities[index, :] = res.intensity
            for k in parameters.keys():
                dsets_meta[k][index] = res.params_nm[k]

        logger.info(f'Starting the generation with {os.cpu_count()} CPU...')
        if max_workers == 0:
            for i, params in enumerate(tqdm(param_grid, total=n_signal, desc="Generating signals", mininterval=1, miniters=100)):
                res = generateSignal(params, q, shape, material_sld, solvent_sld)
                _writeH5(res, i)
        else:
            # Parallel mode
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers,
            ) as executor:
                futures = executor.map(
                    generateSignal,
                    param_grid,
                    repeat(q),
                    repeat(shape),
                    repeat(material_sld),
                    repeat(solvent_sld)
                )
                for i, res in enumerate(tqdm(futures, total=n_signal, desc="Generating signals", mininterval=1, miniters=100)):
                    _writeH5(res, i)

        # Define attrs
        f.create_dataset('q', data=q, dtype=np.float64)
        f.attrs["material"] = material
        f.attrs["technique"] = "saxs"
        f.attrs["techno"] = "sasmodels"
        f.attrs["type"] = "simulation"
        f.attrs["shape"] = shape
        f.attrs["environment"] = env
        f.attrs["scattering_length_density"] = material_sld_cm2
        f.attrs["environment_scattering_length_density"] = solvent_sld_cm2

        for key, value in other_attrs.items():
            f.attrs[key] = value

    logger.info(f"All signals written to {save_h5_filepath}.")

def safePath(path: str, suffix: str = "_", max_try:int = 1000) -> str:
    base, ext = os.path.splitext(path)

    for i in range(max_try):
        candidate = path if i == 0 else f"{base}{suffix}{i}{ext}"
        try:
            with open(candidate, "x"):
                return candidate  # file is now reserved
        except FileExistsError:
            continue

    raise FileExistsError(f"Could not find unique path for '{path}'")

if __name__ == '__main__':
    # py -m kernprof -l -v saxs_simulation_generation.py > report.txt
    # -l line by line -v in the consol > consol display in a txt
    q = np.linspace(1e-3, 1, 1000)
    concentrations = np.logspace(8, 19, 1000).astype(np.float64)
    shape = "cube"
    material = "ag"
    env = "water"
    parameters_operator = 'product'
    save_h5_filepath = rf'C:\Users\ET281306\Downloads\saxs_{material}_{shape}.h5'
    max_workers=0
    other_attrs = {
        "author":"Esteban THEVENON",
        "type":"simulation"
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
