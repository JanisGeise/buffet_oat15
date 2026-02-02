"""
    create coordinates so they can be used in blockMesh
"""
from block_mesh_generator import BlockMeshGenerator


if __name__ == "__main__":
    # name of the file containing the coordinates, it is assumed that there is one line for the header
    file_name = "oat15.dat"

    # path to where the blockMeshDict should be written to
    write_path = r"../OAT15_simulations/URANS_SA_SALSA_validation_Re3e6_Ma0.73/system"

    # execute the blockMeshDict generation with the chosen settings
    grid_generator = BlockMeshGenerator(".", file_name, write_path, reverse=True)
    grid_generator.generate_block_mesh()
    grid_generator.write_block_mesh_dict()
