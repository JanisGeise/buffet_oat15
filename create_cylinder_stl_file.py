"""
    create the STL file for the cylinder used to snap the domain boundary (required for meshing)
"""
import pyvista as pv
from os import makedirs
from os.path import join, exists

if __name__ == "__main__":
    # create the geometry directory if not exists
    if not exists("geometry"):
        makedirs("geometry")

    # create the STL file for the cylinder used to snap the domain boundary
    cylinder = pv.Cylinder(center=[0, -0.1, 0], direction=[0, -1, 0], radius=4.5, height=0.35)
    cylinder.save(join("geometry", "cylinder.stl"))
