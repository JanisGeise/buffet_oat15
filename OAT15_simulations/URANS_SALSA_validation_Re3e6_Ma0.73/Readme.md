# URANS validation case with the SALSA turbulence model

This directory contains the URANS validation case at $Ma_\infty = 0.73$ using the *SALSA* turbulence model.
The solver is `rhoPimpleFoam`, the mesh is generated with `blockMesh` and the case is executed via the `Allrun` script.

- $Re = 3\cdot 10^6$
- $Ma_\infty = 0.73$
- $\alpha = 2.5^\circ, 3.5^\circ$
- $t_\mathrm{end} = 2$ $s~(\approx 484$ $\mathrm{CTU})$

**Note:** since the *SALSA* turbulence model is not directly available in `OpenFOAM`, it has to be compiled from
[here](https://github.com/JanisGeise/OF_SA_SALSA) (see the main `README.md`).

## Validation
In order to validate the simulation setup, the experimental data of *Jacquin et al.* is used (see references). The
extracted pressure distributions as well as $p^\prime_\mathrm{RMS}$ are located as `csv` files in the directory
`validation_exp_data` of this repository.

## Reference
- L. Jacquin, P. Molton, S. Deck, B. Maury, and D. Soulevant. *Experimental Study of Shock Oscillation over a Transonic
Supercritical Profile*, AIAA JOURNAL Vol. 47, No. 9, September 2009, https://arc.aiaa.org/doi/10.2514/1.30190
