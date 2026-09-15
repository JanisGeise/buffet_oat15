# DDES validation case with the SA turbulence model

This directory contains the DDES validation case at $Ma_\infty = 0.73$ using the *Spalart-Allmaras* turbulence model
(`SpalartAllmarasDDES`). The solver is `rhoPimpleFoam` and the case is executed via the `Allrun` script. The base mesh
is generated with `blockMesh` and refined in the wake region with `snappyHexMesh` to obtain a proper DDES mesh.

- $Re = 3\cdot 10^6$
- $Ma_\infty = 0.73$
- $\alpha = 2.5^\circ, 3.5^\circ$
- $t_\mathrm{end} = 0.4$ $s~(\approx 97$ $\mathrm{CTU})$

**Note:** to execute the DDES, `OpenFOAMv2412` or higher is required.

## Validation
In order to validate the simulation setup, the experimental data of *Jacquin et al.* is used (see references). The
extracted pressure distributions as well as $p^\prime_\mathrm{RMS}$ are located as `csv` files in the directory
`validation_exp_data` of this repository.

## Reference
- L. Jacquin, P. Molton, S. Deck, B. Maury, and D. Soulevant. *Experimental Study of Shock Oscillation over a Transonic
Supercritical Profile*, AIAA JOURNAL Vol. 47, No. 9, September 2009, https://arc.aiaa.org/doi/10.2514/1.30190
