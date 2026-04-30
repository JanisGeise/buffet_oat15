"""
script to create directories and modify the inlet boundary condition inn order to execute the parameter study
"""
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
    shutil.copytree(source, destination, dirs_exist_ok=True)

def write_Allrun(target: str) -> None:
    _msg = r"""#!/bin/bash

cd "${0%/*}" || exit                                # Run from this directory
. "${WM_PROJECT_DIR:?}"/bin/tools/RunFunctions      # Tutorial run functions
#------------------------------------------------------------------------------

# execute flow solver
runParallel "$(getApplication)"
"""

    with open(join(target, "Allrun"), "w") as f_out:
        f_out.write(_msg)

def cleanCopy(destination: str) -> None:
    # overwrite the Allrun file
    write_Allrun(destination)

    # clean-up
    _zero_files = glob(join(destination, "processor*", "**", "*_0"), recursive=True)

    # keep forces etc. for now
    # shutil.rmtree(join(destination, "postProcessing"))
    os.remove(join(destination, "log.rhoPimpleFoam"))
    [os.remove(file) for file in _zero_files]

def modify_controlDict(target: str, t_end: Union[float, int] = 1) -> None:
    with open(join(target, "system", "controlDict"), "r") as f_in:
        lines = f_in.readlines()

    # replace the end time
    lines = [line.replace(line, f"endTime\t\t{t_end};\n") if line.startswith("endTime") else line for line in lines]

    # update the control dict
    with open(join(target, "system", "controlDict"), "w") as f_out:
        f_out.writelines(lines)

def relace_all_inlet_conditions(target: str, new_a, new_f, last_write_time = 0.06) -> None:
    for processor in glob(join(target, "processor*", str(last_write_time), "U")):
        replace_frequency_and_amplitude(processor, new_a, new_f)

def replace_frequency_and_amplitude(file_path, a_mod, f_mod) -> None:
    with open(join(file_path), "rb") as f_in:
        lines = f_in.readlines()

    # replace the expression
    idx, val = [(i, l) for i, l in enumerate(lines) if b"valueExpr" in l][0]
    lines[idx] = lines[idx].replace(b"0*sin(2*pi()*0*time()", f"{a_mod}*sin(2*pi()*{f_mod}*time()".encode())

    with open(join(file_path), "wb") as f_out:
        f_out.writelines(lines)

def write_jobscript(target: str, amp, freq) -> None:
    _msg = fr"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=48:00:00
#SBATCH --job-name=A{amp}_f{freq}
#SBATCH --account=p_shockbuffets
#SBATCH --mail-type=start,end\
#SBATCH --mail-user=janis.geise@tu-dresden.de

module load release/24.04 GCC/12.3.0
module load OpenMPI/4.1.5
module load OpenFOAM/v2406
source $FOAM_BASH

# execute
./Allrun &> "log.main"
"""

    with open(join(target, "jobscript"), "w") as f_out:
        f_out.writelines(_msg)

def write_shellscript(all_cases: list, target: str = ".") -> None:
    _msg = ["#!/bin/bash\n\n"]

    # loop over all directories and submit the jobscript
    for d in all_cases:
        _msg.append(f"cd '{d}/'\nsbatch jobscript\ncd ..\n\n")

    # write script
    with open(join(target, "submit_all"), "w") as f_out:
        f_out.writelines(_msg)


def get_last_write_time(base: str) -> float:
    all_times = sorted(glob(join(base, "processor0", "0.*")), key=lambda x: float(x.split("/")[-1].split(".")[-1]))
    return float(all_times[-1].split("/")[-1])


if __name__ == "__main__":
    # TODO: document all functions etc.
    # path to the base case
    BASE_DIR = "base"

    # which amplitudes and frequencies to run
    amplitudes = [0.875, 1.75]
    frequencies = list(range(4, 31))

    # check if we executed the base case, if not exit
    if not os.path.exists(join(BASE_DIR, "log.rhoPimpleFoam")):
        raise FileNotFoundError(f"Base simulation '{BASE_DIR}' must be executed before running the parameter study.")

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

                # clean it up and overwrite the Allrun file
                cleanCopy(_cwd)

                # update the endTime of the simulation
                modify_controlDict(_cwd)


                # loop over all processor directories and insert the correct amplitude and frequency in the inlet
                # boundary condition for the velocity field of the last write time
                last_write_time = get_last_write_time(BASE_DIR)
                relace_all_inlet_conditions(_cwd, a, f, last_write_time=last_write_time)

                # write a jobscript
                write_jobscript(_cwd, a, f)

    # write a shell script, which loops over all simulation directories and submits the jobscript
    write_shellscript(all_dirs)
    logger.info(f"All cases created. Now run '$ . submit_all' run the simulations.")
