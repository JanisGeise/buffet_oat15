"""
    create coordinates so they can be used in blockMesh
"""
from pandas import read_csv
from block_mesh_generator import BlockMeshGenerator



if __name__ == "__main__":
    """
    # reverse the coordinates of the OAT airfoil, so we can use  it in our blockMesh mesher
    oat = read_csv("oat15.dat", header=None, names=["x", "y"], sep=r"\s+")

    with open("oat15_reversed.dat", "w") as f:
        f.write("ONERA OAT15A\n")
        for x, y in zip(reversed(oat["x"]), reversed(oat["y"])):
            f.write(f"{x:.8f}\t{y:.8f}\n")
    """

    file_name = "oat15_reversed.dat"
    write_path = r"../OAT15_simulations/URANS_SA_SALSA_validation_Re3e6_Ma0.73/system"
    grid_generator = BlockMeshGenerator(".", file_name, write_path, expansion_rate=1.2)
    grid_generator.generate_block_mesh()
    grid_generator.write_block_mesh_dict()
