# Transonic shock buffet of an OAT15 airfoil in OpenFOAM
Setup for execution and analysis of simulations of transonic shock buffet for an *ONERA OAT15A* airfoil

**Note:** to execute the DDES, `OpenFOAMv2412` or higher is required.

## TODO

- add DDES setup once URANS setup is validated
- add sketch of the block mesh
- higher sampling rate of surfaces -> avg. cp

## Setup
The meshing of the airfoil is done with `blockMesh`. Therefore, a python script to generate the `blockMeshDict`is available.
To run the simulation follow these steps:

**Meshing:**

1. the meshing tool is located inside the `grid_generation` directory
2. it is expected, that the airfoil coordinates are provided in a textfile within the same directory
3. the coordinates have to be sorted as *TE -> LE via SS -> TE via PS*. There is the option `reverse`, which can be passed
to the `blockMeshGenerator` in case the coordinates are sorted as *TE -> LE via PS -> TE via SS*
4. adjust the path to the desired output directory
5. execute the `generate_grid.py` to generate the `blockMeshDict`

**Simulation:**

1. the simulation setups are located in the directory `OAT15_simulations`
2. run the meshing tool to generate a `blockMeshDict`
3. execute the simulation via the `Allrun` script

**Note:** To achieve buffet, the *SA-SALSA* turbulence model is required. Since this model is not directly available in 
`OpenFOAM` as of now, it has to be compiled from [here](https://github.com/JanisGeise/OF_SA_SALSA).

## Validation
In order to validate the simulation setup, the experimental data of J*acquin et al.* is used (see references).
The extracted pressure distribution for $\alpha = 2.5^\circ, 3.5^\circ$ as well as $p^\prime_\mathrm{RMS}$ is located 
as `csv` files in the directory `validation_exp_data`. The corresponding numerical setup can be found in 
`OAT15_simulations/URANS_SA_SALSA_validation_Re3e6_Ma0.73`. The initial and boundary conditions are as follows:

- $Re = 3\cdot 10^6$
- $Ma_\infty = 0.73$
- $U_\infty = 242.16629 \, m/s$
- $c = 1 \,m$
- $\alpha = 2.5^\circ, 3.5^\circ$
- $t_\mathrm{end} = 1 \,s (\approx 242 \, \mathrm{CTU})$

## Visualization of the results

- the directory `post_pocessing` contains the scripts for post-processing and visualization of the results

## References

**Ma = 0.72, Re = 2e6, $\mathbf{\alpha = 5^\circ}$:**
- J. Kleinert, M. Ehrle, A. Waldmann, and T. Lutz. *Wake Tail Plane Interactions for a Tandem Wing Config-
uration in High-Speed Stall Conditions.* 2023. doi: [10.1007/s13272-023-00670-1](https://link.springer.com/article/10.1007/s13272-023-00670-1). 
- J. Kleinert, J. Stober, and T. Lutz. *Numerical simulation of wake interactions on a tandem wing configuration
in high-speed stall conditions.* In: CEAS Aeronautical Journal 14.1 (2023), pp. 171–186. doi: [10.1007/s13272-022-00634-x](https://link.springer.com/article/10.1007/s13272-022-00634-x)

**Validation at Ma = 0.73, Re = 3e6, $\mathbf{\alpha = 2.5^\circ, 3.5^\circ}$:**
- L. Jacquin, P. Molton, S. Deck, B. Maury, and D. Soulevant. *Experimental Study of Shock Oscillation over a Transonic
Supercritical Profile*, AIAA JOURNAL Vol. 47, No. 9, September 2009, https://arc.aiaa.org/doi/10.2514/1.30190

**Influence of the turbulence model:**
- D.-M. Zimmermann, R. Mayer, T. Lutz, and E. Krämer. *Impact of Model Parameters of SALSA Turbulence Model on 
Transonic Buffet Prediction.* AIAA Journal Vol. 56, No. 2, February 2018, https://arc.aiaa.org/doi/10.2514/1.J056193