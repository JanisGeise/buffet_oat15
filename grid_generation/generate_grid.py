"""
    create coordinates so they can be used in blockMesh
"""
from block_mesh_generator import BlockMeshGenerator


if __name__ == "__main__":
    # name of the file containing the coordinates, it is assumed that there is one line for the header
    file_name = "oat15.dat"

    # path to where the blockMeshDict should be written to
    write_path = r"../OAT15_simulations/DDES_SA_SALSA_validation_Re3e6_Ma0.73/system"

    # execute the blockMeshDict generation with the chosen settings
    # for URANS: n_cells_z=1, y_min=-0.0375, for DDES: n_cells_z=30, y_min=-0.25
    grid_generator = BlockMeshGenerator(".", file_name, write_path, reverse=True, n_cells_z=65, y_min=-0.25)
    grid_generator.generate_block_mesh()
    grid_generator.write_block_mesh_dict()
