"""
script to create directories and modify the inlet boundary condition in order to execute the parameter study
"""
import math

import os
import shutil
import logging

from glob import glob
from os.path import join
from typing import Union

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    force=True)

def createCopies(source: str, destination: str) -> None:
    """
    create copies of the base case

    :param source: path to the base simulation
    :type source: str
    :param destination: path to the target directory
    :type destination: str
    :return: None
    """
    shutil.copytree(source, destination, dirs_exist_ok=True)

def modify_controlDict(target: str, t_end: Union[float, int] = 1.5) -> None:
    """
    change the end time of the simulation in the controlDict

    :param target: top-level path to the simulation directory
    :type target: str
    :param t_end: new end time in seconds, defaults to 1.5s
    :type t_end: Union[float, int]
    :return: None
    """
    with open(join(target, "system", "controlDict"), "r") as f_in:
        lines = f_in.readlines()

    # replace the end time
    lines = [line.replace(line, f"endTime\t\t{t_end};\n") if line.startswith("endTime") else line for line in lines]

    # update the control dict
    with open(join(target, "system", "controlDict"), "w") as f_out:
        f_out.writelines(lines)

def replace_frequency_and_amplitude(file_path: str, a_mod: Union[int, float], f_mod: Union[int, float]) -> None:
    """
    replace the pitching amplitude and frequency of the inlet boundary condition in the U file of a single processor
    directory

    :param file_path: path to the processor directory
    :type file_path: str
    :param a_mod: new pitching amplitude
    :type a_mod: Union[int, float]
    :param f_mod: new pitching frequency
    :type f_mod: Union[int, float]
    :return: None
    """
    with open(join(file_path, "0.orig", "pointDisplacement"), "r") as f_in:
        lines = f_in.readlines()

    # we have to case f in rad/s
    f_mod *= (2*math.pi)

    # replace the expression
    idx, val = [(i, l) for i, l in enumerate(lines) if "            omega       0.0;" in l][0]
    lines[idx] = lines[idx].replace("            omega       0.0;", f"            omega       {f_mod};")
    lines[idx+1] = lines[idx+1].replace("            amplitude   (0 0 0);", f"            amplitude   (0 {a_mod} 0);")

    with open(join(file_path, "0.orig", "pointDisplacement"), "w") as f_out:
        f_out.writelines(lines)

def write_jobscript(target: str, amp: Union[int, float], freq: Union[int, float], wall_time: str = "48:00:00",
                    n_cpu: int = 10) -> None:
    """
    write a jobscript for a simulation setup when executed on an HPC system

    :param target: path to the target simulations folder
    :type target: str
    :param amp: pitching amplitude of the case, used for jobname
    :type amp: Union[int, float]
    :param freq: pitching frequency of the case, used for jobname
    :type freq: Union[int, float]
    :param wall_time: wall time of the job
    :type wall_time: str
    :param n_cpu: number of CPUs to execute the job on
    :type n_cpu: int
    :return: None
    """
    # make sure the job name is reasonable
    amp, freq = "{:.3f}".format(amp), "{:.3f}".format(freq)

    # # SBATCH header
    _msg = fr"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={n_cpu}
#SBATCH --time={wall_time}
#SBATCH --job-name=A{amp}_f{freq}

module load release/24.04 GCC/12.3.0
module load OpenMPI/4.1.5
module load OpenFOAM/v2406
source $FOAM_BASH

# execute
./Allrun &> "log.main"
"""

    with open(join(target, "jobscript"), "w") as f_out:
        f_out.writelines(_msg)

def write_shellscript(all_cases: list, target: str = ".", execution: str = "slurm") -> None:
    """
    write a final shell script which loops over all simulation directories defined and submits the jobscript

    :param all_cases: list containing the top-level directory names of the simulations to submit
    :type all_cases: list
    :param target: target directory to write the shell script to, defaults to cwd
    :type target: str
    :param execution: system used for execution, either 'slurm' or 'local', defaults to 'slurm'
    ::type execution: str
    :return: None
    """
    _msg = ["#!/bin/bash\n\n"]

    # loop over all directories and submit the jobscript
    for d in all_cases:
        if execution.lower() == "slurm":
            _msg.append(f"cd '{d}/'\nsbatch jobscript\ncd ..\n\n")
        else:
            _msg.append(f"cd '{d}/'\n./Allrun\ncd ..\n\n")

    # write script
    with open(join(target, "submit_all"), "w") as f_out:
        f_out.writelines(_msg)


def get_last_write_time(base: str) -> float:
    """
    get the last time folder of the executed base simulation

    :param base: path to the base simulation directory
    :type base: str
    :return: last write time of the base simulation
    :rtype: str
    """
    all_times = sorted(glob(join(base, "processor0", "0.*")), key=lambda x: float(x.split("/")[-1].split(".")[-1]))
    return float(all_times[-1].split("/")[-1])


if __name__ == "__main__":
    # path to the base case
    BASE_DIR = "base"
    SLURM = True
    tend = 0.5

    # which amplitudes to run, here 0.25, 0.35 and 0.50 times the mean cl (without pitching)
    amplitudes = [0.875]

    # which frequencies to run, here: from 0.75 f_buffet to 1.25 f_buffet in 0.5 Hz spacing
    frequencies = list(range(3, 30))
    # frequencies += [i + 0.5 for i in range(10, 30)]
    frequencies = sorted(frequencies)

    # loop over all amplitudes and frequencies and set up the simulations to run
    all_dirs = []
    for a in amplitudes:
        for f in frequencies:
            _cwd = f"A{a}_f{f}"

            # check if we already executed this configuration
            if not os.path.exists(_cwd):
                logger.info(f"Creating case {_cwd}.")
                # save the path
                all_dirs.append(_cwd)

                # copy the base directory
                createCopies(BASE_DIR, _cwd)

                # update the endTime of the simulation
                modify_controlDict(_cwd, t_end=tend)

                # update the mesh motion
                replace_frequency_and_amplitude(_cwd, a, f)

                # write a jobscript if we are on an HPC system
                if SLURM:
                    write_jobscript(_cwd, a, f)
            else:
                logger.info(f"Skipping case {_cwd} since it already exists.")

    # write a shell script, which loops over all simulation directories and submits the jobscript
    write_shellscript(all_dirs, execution="slurm" if SLURM else "local")
    logger.info(f"All cases created. Now run '$ . submit_all' run the simulations.")
