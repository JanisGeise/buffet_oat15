# Transonic shock buffet of an OAT15 airfoil in OpenFOAM
Setup for execution and analysis of simulations of transonic shock buffet for an OAT 15 airfoil

## TODO

- influence tripping
- extend post-processing scripts for analysis of cp-distributions, tau_w, yPlus and shock position
- refactoring and documentation

- higher sampling rate of surfaces
- include sweep angle in IC
- analyse influnce sweep angle
- include pitching for DMDc analysis

## Setup
- the STL file for the OAT15 airfoil has to be located in a directory (*geometry*) in the
top-level of the repository

- the simulation setup is located in the directory `OAT15`
- to execute the meshing run the `Allrun.pre`
- to execute the simulation (including the meshing) run the `Allrun` script

## Visualization of the results

- the directory `post_pocessing` contains the scripts for post-processing and visualization of the results
  (currently only analysis of the force coefficients)

## References

- J. Kleinert, M. Ehrle, A. Waldmann, and T. Lutz. Wake Tail Plane Interactions for a Tandem Wing Config-
uration in High-Speed Stall Conditions. 2023. doi: 10.1007/s13272-023-00670-1. 
- J. Kleinert, J. Stober, and T. Lutz. “Numerical simulation of wake interactions on a tandem wing configuration
in high-speed stall conditions”. In: CEAS Aeronautical Journal 14.1 (2023), pp. 171–186. doi: 10 . 1007 /
s13272-022-00634-x.
