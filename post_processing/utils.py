"""
    helper functions
"""
import numpy as np
import pandas as pd
import torch as pt

from glob import glob
from os.path import join
from scipy.signal import welch
from typing import Union, Tuple
from scipy.interpolate import interp1d
from flowtorch.data import FOAMDataloader


def load_ratio_rans_les(load_path, usecols=[0, 1, 2], names=["t", "les", "rans"]) -> pd.DataFrame:
    dirs = sorted(glob(join(load_path, "postProcessing", "DESField", "*")), key=lambda x: float(x.split("/")[-1]))
    _ratio = [pd.read_csv(join(p, "DESModelRegions.dat"), sep=r"\s+", comment="#", header=None, usecols=usecols,
                          names=names) for p in dirs]

    if len(_ratio) == 1:
        _ratio = _ratio[0]
    else:
        _ratio = pd.concat(_ratio)

    # remove duplicates (resulting from dt < write precision) and reset the idx
    _ratio.drop_duplicates(["t"], inplace=True)
    _ratio.reset_index(inplace=True, drop=True)
    return _ratio


def load_yplus(load_path, patch_name: str = "airfoil") -> pd.DataFrame:
    dirs = sorted(glob(join(load_path, "postProcessing", "yPlus", "*")), key=lambda x: float(x.split("/")[-1]))
    _yplus = [pd.read_csv(join(p, "yPlus.dat"), sep=r"\s+", comment="#", header=None, usecols=list(range(5)),
                          names=["t", "patch", "yPlus_min", "yPlus_max", "yPlus_avg"]) for p in dirs]

    if len(_yplus) == 1:
        _yplus = _yplus[0]
    else:
        _yplus = pd.concat(_yplus)

    # only keep the target patch name
    _yplus = _yplus[_yplus.patch == patch_name]

    # remove duplicates (resulting from dt < write precision) and reset the idx
    _yplus.drop_duplicates(["t"], inplace=True)
    _yplus.reset_index(inplace=True, drop=True)

    return _yplus


def load_force_coeffs(load_path, usecols=[0, 1, 4], names=["t", "cx", "cy"]) -> pd.DataFrame:
    dirs = sorted(glob(join(load_path, "postProcessing", "forces", "*")), key=lambda x: float(x.split("/")[-1]))
    coeffs = [pd.read_csv(join(p, "coefficient.dat"), sep=r"\s+", comment="#", header=None, usecols=usecols, names=names)
              for p in dirs]

    if len(coeffs) == 1:
        coeffs = coeffs[0]
    else:
        coeffs = pd.concat(coeffs)

    # remove duplicates (resulting from dt < write precision) and reset the idx
    coeffs.drop_duplicates(["t"], inplace=True)
    coeffs.reset_index(inplace=True, drop=True)
    return coeffs


def compute_fft(data: np.ndarray, dt: Union[float, int]) -> Tuple[np.ndarray, np.ndarray]:
    _f, _a = welch(data, 1/dt, nperseg=len(data), nfft=len(data), window="boxcar")
    return _f, _a


def interpolate_uniform(t: np.ndarray, data: np.ndarray):
    # get start and end time
    t_start, t_end = t[0], t[-1]

    # use standard interpolation to get values at const. dt
    _interpolator = interp1d(t, data, fill_value="extrapolate")
    dt = float("{:.1e}".format(t[-1] - t[-2]))

    t_new = np.arange(start=t_start + dt, stop=t_end, step=dt)
    return t_new, _interpolator(t_new)


def get_pimple_iterations(load_path: str) -> Tuple[list, list]:
    """
    gets the number of PIMPLE iterations (p-U-couplings)

    :param load_path: path to the top-level directory of the simulation containing the log file from the flow solver
    :return: dict containing the mean and max. Courant numbers, and if present the mean and max. CFL from the interface
    """
    pattern = [r"PIMPLE: not converged within ", r"PIMPLE: converged in "]

    # check if we have multiple log files, if so sort them
    try:
        logs = sorted(glob(join(load_path, f"log.rhoPimpleFoam*")), key=lambda x: int(x.split("_")[-1]))
    except ValueError:
        logs = glob(join(load_path, f"log.rhoPimpleFoam*"))

    data, times = [], []
    for log in logs:
        with open(log, "r") as f:
            logfile = f.readlines()

        for line in logfile:
            if line.startswith(pattern[0]) or line.startswith(pattern[1]):
                data.append(int(line.split(" ")[-2]))
            elif line.startswith("Time = "):
                times.append(float(line.split()[-1].strip()))
    return times, data


def compute_norm_of_fields(load_path: str, time_boundaries: list = None,
                           field: str = "UMean") -> Tuple[pt.Tensor, list]:
    """
    TODO: doku

    :param load_path:
    :param time_boundaries:
    :param field:
    :return:
    """
    print(f"Starting with case: {load_path}.")
    loader = FOAMDataloader(load_path)

    # get the defined boundaries for start and end time to use if provided
    if time_boundaries is not None:
        idx = sorted([i for i, t in enumerate(loader.write_times) if t in time_boundaries])
        write_times = loader.write_times[idx[0]:idx[1]+1]

    # else use all times steps but zero
    else:
        write_times = loader.write_times[1:]

    # check for the time steps in which the target filed is present
    write_times = [t for t in write_times if field in loader.field_names[t]]

    # compute the norm of the field in the last time step
    norm_first_field = loader.load_snapshot(field, write_times[0]).norm()

    # now compute the difference of the norm between two consecutive time steps
    all_norms, last_snapshot = [], 0
    for i in range(len(write_times)):
        print(f"Loading time step ({i+1} / {len(write_times)}) t = {write_times[i]} s.")
        if i == 0:
            last_snapshot = loader.load_snapshot(field, write_times[i])
            continue
        new_snapshot = loader.load_snapshot(field, write_times[i])
        dt = float(write_times[i]) - float(write_times[i-1])
        all_norms.append(((new_snapshot-last_snapshot) / dt).norm() / norm_first_field)
        last_snapshot = new_snapshot

    # don't return the norm of the last field, since the difference is zero
    return pt.tensor(list(map(float, write_times)))[1:], all_norms


if __name__ == "__main__":
    pass
