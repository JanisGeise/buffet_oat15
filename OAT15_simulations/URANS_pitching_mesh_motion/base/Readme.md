# Guide for running the URANS simulations with pitching via mesh motion

This directory contains the setup and utilities to execute the URANS with pitching instead of a constant angle of attack.
The URANS simulation (directory *base*) corresponds to the setup of `URANS_SALSA_validation_Re3e6_Ma0.73`. In contrast
to the setup in `URANS_pitching`, the pitching motion is prescribed via mesh motion of the airfoil using an
`oscillatingRotatingMotion` about the quarter-chord point. The amplitude is given in $[^\circ]$ about the $y$-axis and
the circular frequency $\omega = 2\pi f$ with $f$ in $[Hz]$, while the deformation of the mesh is computed with a
`displacementLaplacian` solver. This setup is validated against the experimental data of *Jacquin et al.* (see
references below), but may become unstable for high pitching amplitudes.

## Setting up the base case
The pitching amplitude and frequency are prescribed in the `airfoil` boundary condition of the `0.orig/pointDisplacement`
file, where `amplitude` is given in $[^\circ]$ and `omega` in $[rad/s]$. By default, both are set to zero, such that the
base case is a steady inflow case with $\alpha = 3.5^\circ$. To change the inflow angle or inflow velocity, the entries
`Uinlet` and `alpha0` in the `0.orig/U` file have to be adjusted.

By default, the end time of the simulation is set to $t = 1$ $s$. To execute the base case, just run the `Allrun` script.

## Setting up the parameter study
Analogous to the setup in `URANS_pitching`, the script `run_parameter_study.py` contains three modifiable entries

- SLURM: flag for executing the parameter study on an HPC system
- tend: physical end time of the simulations
- amplitudes: list containing the pitching amplitudes to run (in degree)
- frequencies: list containing the pitching frequencies to run (in Hz)

The script loops over each amplitude-frequency combination, copies the base case, modifies the `pointDisplacement`
boundary condition, adjusts the end time in the `controlDict` and writes a jobscript (if `SLURM` is set). Afterwards,
the simulations can be executed via the generated `submit_all` script. Note that the `SBATCH` settings have to be
adjusted when executing on an HPC system (function `write_jobscript`).

## Checking the mesh motion
The `AllrunMeshMotion` script executes a pure mesh motion without solving the flow (`moveDynamicMesh`). It can be used
to check the mesh quality during the deformation. The start and end time, the time step and the write interval are
controlled via the environment variables `MESH_MOTION_START_TIME`, `MESH_MOTION_END_TIME`, `MESH_MOTION_DELTA_T` and
`MESH_MOTION_WRITE_INTERVAL`, e.g.

```
MESH_MOTION_END_TIME=0.1 MESH_MOTION_DELTA_T=1e-4 ./AllrunMeshMotion
```

## Validation
In order to validate the simulation setup, the experimental data of *Jacquin et al.* is used. The extracted pressure
distributions as well as $p^\prime_\mathrm{RMS}$ are located as `csv` files in the directory `validation_exp_data` of
this repository.

## Post-processing tools

Some Jupyter notebooks to post-process the simulations, e.g. to analyze synchroniztion, are provided in the directory
`post_processing` of this repository.

## Reference:

Experimental Study of Shock Oscillation over a Transonic Supercritical Profile, L. Jacquin,∗ P. Molton,† S. Deck,‡ B. Maury,† and D. Soulevant, https://arc.aiaa.org/doi/10.2514/1.30190
