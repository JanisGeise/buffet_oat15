# Guide for running the URANS simulations with pitching

This directory contains the setup and utilities to execute the URANS with pitching instead of a constant angle of attack.
The URANS simulation (directory *base*) is the same as `URANS_SALSA_validation_Re3e6_Ma0.73`.
The pitching motion is prescribed via a periodically varying inflow angle as

$\alpha(t) = \alpha_0 + A \sin(\omega t - \varphi)$

with $\alpha_0$ denoting the mean angle of attack, $A$ the amplitude in $[^\circ]$, $\varphi$ the phase and the circular 
frequency $\omega = 2\pi f$ with $f$ in $[Hz]$. To decrease the simulation runtime, first, a base case with 
$\alpha = \alpha_0 = const.$ is executed. Then the `run_parameter_study.py` script can be executed to create copies of the base
case and running each of these simulations with a different $f$ and $A$. The following sections will provide some guidance
on how to execute the parameter study.

## Setting up the base case
The base case is by default set up with a mean inflow angle of $\alpha_0 = 3.5^\circ$.
To change the inflow angle or inflow velocity, **both** entries for `Uinlet` and `alpha0` (at the top and inside the `inlet_outlet`
boundary condition) have to be adjusted.
**It is important that the `valueExpr` in the `base/0.orig/U` file remains unchanged.** Otherwise, the execution 
of the parameter study will not work.

By default, the end time of the simulation is set to $t = 0.06$s, which covers the initial transient until the flow develops.
To execute the base case, just run the `Allrun` script.

## Setting up the parameter study
Once the base case is completed, we can set up the parameter study.
The script `run_parameter_study` contains three modifiable entries

- SLURM: flag for executing the parameter study on an HPC system
- tend: physical end time of the simulations, defaults to $2$s
- amplitudes: list containing the pitching amplitudes to run (in degree)
- frequencies: list containing the pitching frequencies to run (in Hz)

To execute the script, just run `python3 runparameter_study.py`. The script will then loop over each amplitude-frequency
combination, copy the base case, modify the boundary condition and make some other adjustments.

Note that the `SBATCH` settings have to be adjusted when executing on an HPC system (function `write_jobscript`).

## Running the parameter study
Once the parameter study is set up, we can execute the simulations.
To execute the simulations just run the `submit_all` bash script as `$. submit_all` (on both an HPC system or local machine).

## Post-processing tools

Some Jupyter notebooks to post-process the simulations, e.g. to analyze synchroniztion, are provided in the directory
`post_processing` of this repository.