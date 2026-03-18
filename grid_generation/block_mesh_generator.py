"""
class for generating a blockMesh based on 2D coordinates of an airfoil
"""
import torch as pt
import numpy as np

from os.path import join
from copy import deepcopy
from pandas import read_csv
from typing import Union, Tuple, List
from scipy.interpolate import interp1d

HEADER = r"""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  2412                                  |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

class BlockMeshGenerator:
    def __init__(self, load_path: str, file_name: str, write_path: str, scale: Union[int, float] = 1,
                 y_min: Union[int, float] = -0.0375, y_max: Union[int, float]=0, n_cells_z: int = 1,
                 distance_LE: Union[int, float] = 0.05, first_layer_thickness: Union[int, float]=1e-5, reverse: bool = False,
                 use_yPlus: bool = False, expansion_rate: Union[int, float] = 1.2, projection_distance: list = None,
                 reynolds_number: Union[int, float] = None, yPlus_target: float = 1, u_infinity: Union[int, float] = 1,
                 density: Union[int, float] = 1, round_TE: float = 0.998):
        """
        Implements a class for generating a blockMeshDict based on airfoil coordinates. The resulting mesh will be
        oriented in the x-z-plane while they y-direction marks the spanwise direction.

        :param load_path: path to the file containing the airfoil coordinates
        :param file_name: name of the file containing the coordinates
        :param write_path: path to where the blockMeshDict should be written to
        :param scale: ratio to which the airfoil should be scaled to, defaults to 1
        :param y_min: min. extrusion coordinate, defaults to y = -0.0375
        :param y_max: max. extrusion coordinate, defaults to y = 0
        :param n_cells_z: number of cells in spanwise direction, defaults to 1 (URANS)
        :param distance_LE: relative distance of the chord length in x-direction after which the first block should be placed
        :param first_layer_thickness: height of the first cell
        :param reverse: reverse the airfoil coordinates in case the orientation is TE -> LE (via PS) -> TE (via SS)
        :param use_yPlus: use a flat plate computation of y+ to get the approximate first layer height based of the flow
        :param expansion_rate: expansion rate in the boundary layer
        :param projection_distance: distances in the far field for which the different meshes will be created,
                                    relative to the chord length
        :param reynolds_number: Reynolds number if 'use_yPlus' = True
        :param yPlus_target: target y+ to be achieved if 'use_yPlus' = True, defaults to 1
        :param u_infinity: free stream velocity if 'use_yPlus' = True
        :param density: free stream density if 'use_yPlus' = True
        :param round_TE: blending distance to create a rounded trailing edge instead of a blunt one, relative to the chord length.
                        This parameter should lie in the range of 0.995 <= round_TE < 1.0
                        Use with caution.
        """

        # set default distances
        if projection_distance is None:
            projection_distance = [0.01, 0.85, 1.5, 50]
        else:
            # make sure the distances are in ascending order
            projection_distance = sorted(projection_distance)

        # check if the distances / user input are correct
        assert len(projection_distance) == 4, "The argument 'projection_distance' has to be a list containing 4 entries."
        assert len(set(projection_distance)) == 4, "The projection_distances have to be unique."
        assert not (use_yPlus and reynolds_number is None), "Reynolds number must be provided when 'use_yPlus' is 'True'"
        assert 0.995 <= round_TE < 1.0, "The blending factor for the trailing edge has to be 0.995 < round_TE < 1.0"

        # paths and file names
        self._load_path = load_path
        self._write_path = write_path
        self._file_name = file_name

        # flow / domain properties
        self._y_bound = (y_min, y_max)
        self._use_yPlus = use_yPlus
        self._re = reynolds_number
        self._u_infinity = u_infinity
        self._density = density
        self._yPlus_target = yPlus_target

        # number of cells in spanwise direction
        self._nz = n_cells_z

        # airfoil properties
        self._ss = None
        self._ps = None
        self._chord = None
        self._reverse = reverse
        self._rTE = round_TE

        # distances for the projection of the airfoil
        self._grading_normal = []
        self._ss_projected = None
        self._ps_projected = None
        self._n_ss = None
        self._n_ps = None
        self._distance_LE = distance_LE
        self._expansion_rate = [expansion_rate, 1.02, 1.05, 1.05]
        self._distance_projection = projection_distance
        self._first_layer_thickness = [first_layer_thickness]
        self._last_cell_thickness = []

        # nodes dict
        self._all_nodes = {}
        self._all_edges = {}
        self._all_arcs = {}
        self._n_cells = []

        # blockMeshDict
        self._header = HEADER
        self._scale = f"\nscale {scale};\n\n"
        self._footer = "\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
        self._block_mesh_name = "blockMeshDict"
        self._dec = ".8f"
        self._init_edges = False

    def generate_block_mesh(self) -> None:
        """
        Wrapper method to generate the blockMeshDict. It contains the steps:

            - reading the airfoil coordinates
            - computing the gradings and number of blocks
            - compute the node positions
            - write the blockMeshDict

        :return: None
        :rtype: None
        """
        self._read_coordinates()
        self._compute_n_blocks_and_grading_normal()
        self._project_coordinates()
        self._assemble_nodes()

    def write_block_mesh_dict(self) -> None:
        """
        Write all the required entries to a blockMeshDict file.

        :return: None
        :rtype: None
        """
        self._write_header()
        self._write_nodes(self._all_nodes, completed=True)
        self._write_blocks()

        # write arcs for TE (SS)
        self._write_arcs([(33, 12), (37, 14)], self._all_arcs["TE_ss"])
        self._write_arcs([(13, 32), (15, 36)], self._all_arcs["projection_ss"])

        # write arcs for TE (PS)
        self._write_arcs([(30, 17), (34, 19)], self._all_arcs["TE_ps"])
        self._write_arcs([(31, 16), (35, 18)], self._all_arcs["projection_ps"])

        # write arcs for projection
        self._write_arcs([(26, 39), (27, 41)], self._all_arcs["block6plus"])
        self._write_arcs([(28, 38), (29, 40)], self._all_arcs["block6minus"])
        self._write_arcs([(54, 62), (56, 64)], self._all_arcs["block10plus"])
        self._write_arcs([(55, 63), (57, 65)], self._all_arcs["block10plusProj"])
        self._write_arcs([(59, 67), (61, 69)], self._all_arcs["block10minus"])
        self._write_arcs([(58, 66), (60, 68)], self._all_arcs["block10minusProj"])

        # arcs for TE
        self._write_arcs([(32, 31), (36, 35)], self._all_arcs["TE_wake_block4"])
        self._write_arcs([(39, 38), (41, 40)], self._all_arcs["TE_wake_block7"])
        self._write_arcs([(62, 67), (64, 69)], self._all_arcs["block11"])
        self._write_arcs([(63, 66), (65, 68)], self._all_arcs["block11proj"])
        self._write_arcs([(90, 94), (92, 96)], self._all_arcs["block15"])
        self._write_arcs([(91, 95), (93, 97)], self._all_arcs["block15proj"])

        # arc for blocks 14
        self._write_arcs([(82, 90), (84, 92)], self._all_arcs["block14plus"])
        self._write_arcs([(83, 91), (85, 93)], self._all_arcs["block14plusProj"])
        self._write_arcs([(87, 94), (89, 96)], self._all_arcs["block14minus"])
        self._write_arcs([(86, 95), (88, 97)], self._all_arcs["block14minusProj"])

        # block 0
        self._write_edges([(2, 1), (6, 5)], self._all_edges["1_2"])
        self._write_edges([(3, 0), (7, 4)], self._all_edges["3_0"])
        self._write_edges([(1, 9), (5, 11)], self._all_edges["9_1"])
        self._write_edges([(0, 8), (4, 10)], self._all_edges["8_0"])

        # block 1
        self._write_edges([(12, 2), (14, 6)], self._all_edges["2_12"])
        self._write_edges([(13, 3), (15, 7)], self._all_edges["13_3"])
        self._write_edges([(8, 16), (10, 18)], self._all_edges["8_16"])
        self._write_edges([(9, 17), (11, 19)], self._all_edges["17_9"])

        # block 2
        self._write_edges([(21, 20), (23, 22)], self._all_edges["20_21"])
        self._write_edges([(20, 24), (22, 25)], self._all_edges["20_24"])

        # block 3
        self._write_edges([(26, 21), (27, 23)], self._all_edges["21_26"])

        # block 8
        self._write_edges([(44, 43), (48, 47)], self._all_edges["43_44"])
        self._write_edges([(45, 42), (49, 46)], self._all_edges["42_45"])
        self._write_edges([(42, 50), (46, 52)], self._all_edges["42_50"])
        self._write_edges([(43, 51), (47, 53)], self._all_edges["43_51"])

        # block 9
        self._write_edges([(54, 44), (56, 48)], self._all_edges["44_54"])
        self._write_edges([(55, 45), (57, 49)], self._all_edges["45_55"])

        # block 12
        self._write_edges([(72, 71), (76, 75)], self._all_edges["71_72"])
        self._write_edges([(73, 70), (77, 74)], self._all_edges["70_73"])
        self._write_edges([(70, 78), (74, 80)], self._all_edges["70_78"])
        self._write_edges([(71, 79), (75, 81)], self._all_edges["71_79"])

        # block 13
        self._write_edges([(82, 72), (84, 76)], self._all_edges["72_82"])
        self._write_edges([(83, 73), (85, 77)], self._all_edges["73_83"], completed=True)

        self._write_boundaries()
        self._write_merge_patch_pairs()
        self._write_footer()

    def _read_coordinates(self) -> None:
        """
        Read the airfoil coordinates from the provided file. Compute the chord length and split up the coordinates into
        SS and PS.

        :return: None
        :rtype: None
        """
        _coordinates = pt.from_numpy(read_csv(join(self._load_path, self._file_name), sep=r"\s+", skiprows=1,
                                              header=None, names=["x", "y"]).to_numpy())

        # reverse coordinates in case the ordering is the other way round
        if self._reverse:
            _coordinates = reversed(_coordinates)
        self._chord = _coordinates[:, 0].max().item() - _coordinates[:, 0].min().item()

        # split into SS and PS
        LE_idx = pt.where(_coordinates[:, 0] == _coordinates[:, 0].min())[0]
        self._ss = _coordinates[:LE_idx+1]
        self._ps =  _coordinates[LE_idx:]

    def _compute_n_blocks_and_grading_normal(self) -> None:
        """
        Compute the number of cells normal to the airfoil surface for each projection distance.

        :return: None
        :rtype: None
        """
        for i in range(len(self._distance_projection)):
            if i == 0:
                if self._use_yPlus:
                    # alternatively compute number of cells and grading for a given inflow condition and yPlus
                    cf = 0.026 * self._re ** (-1.0 / 7.0)
                    tau_wall = 0.5 * cf * self._density * self._u_infinity ** 2
                    u_tau = pt.sqrt(pt.tensor(tau_wall) / self._density)
                    mu = self._density * self._chord * self._u_infinity / self._re
                    first_layer_thk = (mu * self._yPlus_target / (self._density * u_tau)).item()
                else:
                    first_layer_thk = self._first_layer_thickness[0]
                _dist = self._distance_projection[i] * self._chord
            else:
                # for subsequent blocks use the last cell thickness of the previous block to ensure continuity
                first_layer_thk = self._last_cell_thickness[-1]
                _dist = (self._distance_projection[i] - self._distance_projection[i-1]) * self._chord

            # expansion rate for the current block (geometric growth factor)
            _e = pt.tensor(self._expansion_rate[i])
            if abs(_e - 1.0) < 1e-12:
                # uniform spacing for (expansion rate = 1):
                n_cells = pt.ceil(pt.tensor(_dist / first_layer_thk))
                _thickness_new = n_cells * first_layer_thk
                _grading = pt.ones(1,)
            else:
                # if expansion rate != 1: solve geometric series:
                # dist = first_layer_thk * (e^n - 1) / (e - 1) for n, then take ceiling to ensure full coverage
                n_cells = pt.ceil(pt.log(pt.ones(1,) + (_e - pt.ones(1,)) * _dist / first_layer_thk) / pt.log(_e))

                # total block thickness reconstructed from geometric series
                _thickness_new = first_layer_thk * (_e **n_cells - 1.0) / (_e - 1.0)

                # grading factor for the last cell relative to the first
                _grading = _e**(n_cells - 1)

            # store the computed cells and gradings
            self._grading_normal.append(_grading.item())
            self._n_cells.append(int(n_cells.item()))
            if i == 0:
                self._distance_projection[i] = _thickness_new.item() / self._chord
            else:
                self._distance_projection[i] = (_thickness_new.item() + self._distance_projection[i-1]) * self._chord
            self._last_cell_thickness.append((first_layer_thk * _e**(n_cells - 1)).item())

    def _project_coordinates(self) -> None:
        """
        Compute a projection of the airfoil coordinates in normal direction.

        :return: None
        :rtype: None
        """
        self._get_normal_vectors()
        self._ss_projected = self._compute_projection(self._distance_projection[0] * self._chord, True)
        self._ps_projected = self._compute_projection(self._distance_projection[0] * self._chord, False)

    @staticmethod
    def _compute_normals(coordinates: pt.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the norma vector for each airfoil coordinate in order to compute the projection.

        :param coordinates: coordinates for which to compute the vectors'
        :type coordinates: pt.Tensor
        :return: tangential and normal vectors
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        # compute tangent vectors
        diffs = np.gradient(coordinates, axis=0)
        dx, dy = diffs[:, 0], diffs[:, 1]

        # normalize tangent vectors
        tangent_norm = np.sqrt(dx**2 + dy**2)
        tx, ty = dx / tangent_norm, dy / tangent_norm

        # compute normal vectors
        nx, ny = ty, -tx

        return nx, ny

    def _get_normal_vectors(self) -> None:
        """
        Get and store the normal vectors for both SS and PS.

        :return: None
        :rtype: None
        """
        self._n_ss = self._compute_normals(self._ss)
        self._n_ps = self._compute_normals(self._ps)

    def _compute_projection(self, offset, ss) -> pt.Tensor:
        """
        Compute the projection based on an offset.

        :param offset: offset (distance of the projection)
        :type offset: float
        :param ss: flag if the coordinates belong to the SS (True) or  PS (False)
        :return: Projected coordinates'
        :rtype: pt.Tensor
        """
        if ss:
            return pt.stack([self._ss[:, 0] + offset * self._n_ss[0], self._ss[:, 1] + offset * self._n_ss[1]]).transpose(-1, 0)
        else:
            return pt.stack([self._ps[:, 0] + offset * self._n_ps[0], self._ps[:, 1] + offset * self._n_ps[1]]).transpose(-1, 0)

    def _assemble_nodes(self) -> None:
        """
        Compute the node positions of the mesh and all edges / arc connecting them.

        :return: None
        :rtype: None
        """
        # ------------------------------------- 1st projection inner mesh -------------------------------------
        x_block_0 = pt.where(pt.isclose(self._ss[:, 0], self._ss[-1, 0] + self._distance_LE * self._chord, atol=0.01))[0]

        # exception handling in case of multiple findings
        if x_block_0.size(0) > 1:
            x_block_0 = x_block_0[0]
        # the order for PS is reversed, so we need to reverse the idx as well
        x_block_0neg = self._ps_projected.size(0) - x_block_0

        # edges block 0
        self._all_edges = {"1_2": self._ss[x_block_0:, :], "9_1": self._ps[:x_block_0neg, :],
                                     "3_0": self._ss_projected[x_block_0:, :], "8_0": self._ps_projected[:x_block_0neg, :]}

        # block 0+
        self._all_nodes = {"0": self._ss_projected[-1, :].squeeze(), "1": self._ss[-1, :].squeeze(),
                                     "2": self._all_edges["1_2"][0, :].squeeze(),
                                     "3": self._all_edges["3_0"][0, :].squeeze()}
        # add 3rd dim for front
        for key in self._all_nodes.keys():
            self._all_nodes[key] = pt.cat([self._all_nodes[key], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._create_back_nodes(4, 4)

        # block 0-
        self._all_nodes["8"] = self._all_edges["8_0"][-1, :].squeeze()
        self._all_nodes["9"] = self._all_edges["9_1"][-1, :].squeeze()

        # add 3rd dim for front
        for key in ["8", "9"]:
            self._all_nodes[key] = pt.cat([self._all_nodes[key], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._create_back_nodes(10, 2)

        # interpolate SS airfoil TE
        inter = interp1d(self._ss[:, 0], self._ss[:, 1])
        y_coord_ss = inter(self._rTE * self._ss[0, 0])

        # interpolate projection SS airfoil TE
        inter = interp1d(self._ss_projected[:, 0], self._ss_projected[:, 1])
        y_coord_ss_pro = inter(self._rTE * self._ss_projected[0, 0])

        self._all_nodes["12"] = pt.cat([self._rTE * self._ss[0, 0].unsqueeze(-1), pt.from_numpy(y_coord_ss).unsqueeze(-1)],
                                       dim=-1)
        self._all_nodes["13"] = pt.cat([self._rTE * self._ss_projected[0, 0].unsqueeze(-1),
                                        pt.from_numpy(y_coord_ss_pro).unsqueeze(-1)], dim=-1)

        # add 3rd dim for front
        for key in ["12", "13"]:
            self._all_nodes[key] = pt.cat([self._all_nodes[key], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._create_back_nodes(14, 2)

        # crop airfoil coordinates based on y_coord_ss & y_coord_ss_proj
        idx_ss = pt.where(self._ss[:, 0] < self._rTE * self._ss[0, 0])[0][0].item()
        self._all_edges["2_12"] = self._ss[idx_ss:x_block_0, :]
        self._all_edges["13_3"] = self._ss_projected[idx_ss:x_block_0, :]

        # block 1-
        # interpolate PS airfoil TE
        inter = interp1d(self._ps[:, 0], self._ps[:, 1])
        y_coord_ps = inter(self._rTE * self._ps[-1, 0])

        # interpolate projection SS airfoil TE
        inter = interp1d(self._ps_projected[:, 0], self._ps_projected[:, 1])
        y_coord_ps_pro = inter(self._rTE * self._ps_projected[-1, 0])

        self._all_nodes["16"] = pt.cat([self._rTE * self._ps_projected[-1, 0].unsqueeze(-1),
                                        pt.from_numpy(y_coord_ps_pro).unsqueeze(-1)], dim=-1)
        self._all_nodes["17"] = pt.cat([self._rTE * self._ps[-1, 0].unsqueeze(-1),
                                        pt.from_numpy(y_coord_ps).unsqueeze(-1)], dim=-1)
        for key in ["16", "17"]:
            self._all_nodes[key] = pt.cat([self._all_nodes[key], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._create_back_nodes(18, 2)

        # crop airfoil coordinates based on y_coord_ss & y_coord_ss_proj
        idx_ps = pt.where(self._ps[:, 0] < self._rTE * self._ps[-1, 0])[0][-1].item()

        self._all_edges["8_16"] = self._ps_projected[x_block_0neg:idx_ps, :]
        self._all_edges["17_9"] = self._ps[x_block_0neg:idx_ps, :]

        # ----------------------------------------- 2nd projection inner mesh -----------------------------------------
        # block 2
        new_coord_ss = self._compute_projection(self._distance_projection[1] * self._chord, ss=True)
        new_coord_ps = self._compute_projection(self._distance_projection[1] * self._chord, ss=False)

        # block 2+
        self._all_nodes["20"] = new_coord_ss[-1, :].squeeze()
        self._all_nodes["21"] =  new_coord_ss[x_block_0, :].squeeze()

        for key in ["20", "21"]:
            self._all_nodes[key] = pt.cat([self._all_nodes[key], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._create_back_nodes(22, 2)
        self._all_edges["20_21"] = new_coord_ss[x_block_0:, :]

        # block 2-
        self._all_nodes["24"] = new_coord_ps[x_block_0neg, :].squeeze()
        for key in ["24"]:
            self._all_nodes[key] = pt.cat([self._all_nodes[key], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._create_back_nodes(25, 1)
        self._all_edges["20_24"] = new_coord_ps[:x_block_0neg, :].squeeze()

        # block 3+
        # interpolate projection SS airfoil
        inter = interp1d(new_coord_ss[:, 0], new_coord_ss[:, 1])
        y_coord_ss_pro2 = inter(self._rTE * new_coord_ss[0, 0])

        # interpolate projection PS airfoil
        inter = interp1d(new_coord_ps[:, 0], new_coord_ps[:, 1])
        y_coord_ps_pro2 = inter(self._rTE * new_coord_ps[-1, 0])

        self._all_nodes["26"] = pt.cat([self._rTE * new_coord_ss[0, 0].unsqueeze(-1),
                                        pt.from_numpy(y_coord_ss_pro2).unsqueeze(-1)], dim=-1)
        for key in ["26"]:
            self._all_nodes[key] = pt.cat([self._all_nodes[key], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._create_back_nodes(27, 1)
        self._all_edges["21_26"] = new_coord_ss[idx_ss:x_block_0, :]

        # block 3-
        self._all_nodes["28"] = pt.cat([self._rTE * new_coord_ps[-1, 0].unsqueeze(-1),
                                        pt.from_numpy(y_coord_ps_pro2).unsqueeze(-1)], dim=-1)

        for key in ["28"]:
            self._all_nodes[key] = pt.cat([self._all_nodes[key], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._create_back_nodes(29, 1)

        # now get the gradient of the last point to continue the block with the same curvature, orient on SS since more
        # important
        _grad_ss = (self._ss[0, 1] - self._ss[1, 1]) / (self._ss[0, 0] - self._ss[1, 0])
        # _grad_ps = (self._ps[-1, 1] - self._ps[-2, 1]) / (self._ps[-1, 0] - self._ps[-2, 0])

        # block 4 at TE
        # move the nodes by the same distance in y at TE as we did on the airfoil's surface
        y_pos_TE_PS = self._ps[-1, 1] + (1-self._rTE) * self._chord
        y_pos_TE_SS = self._ss[0, 1] - (1-self._rTE) * self._chord

        # TODO: add assertion in case self._rTE leads to negative block 4
        if y_pos_TE_SS < y_pos_TE_PS:
            raise ArithmeticError

        self._all_nodes["30"] = pt.cat([self._ps[-1, 0].unsqueeze(-1), y_pos_TE_PS.unsqueeze(-1),
                                        pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._all_nodes["31"] = deepcopy(self._all_nodes["17"])
        self._all_nodes["31"][0] += self._distance_projection[0] * self._chord
        self._all_nodes["31"][1] += self._distance_projection[0] * self._chord * _grad_ss * 0.6    # absicht!
        self._all_nodes["32"] = deepcopy(self._all_nodes["12"])
        self._all_nodes["32"][0] += self._distance_projection[0] * self._chord
        self._all_nodes["32"][1] -= self._distance_projection[0] * self._chord * _grad_ss * 0.6     # absicht!

        # block 5+
        self._all_nodes["33"] = pt.cat([self._ss[0, 0].unsqueeze(-1), y_pos_TE_SS.unsqueeze(-1),
                                        pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._create_back_nodes(34, 4)

        # compute point for arc at TE
        self._interpolate_arc_point(self._ss, idx_ss, "33", self._ss[0, 0], "TE_ss")

        # apply same procedure for remaining arcs
        self._interpolate_arc_point(self._ss_projected, idx_ss, "32", self._ss_projected[0, 0], "projection_ss")

        # block 5-
        self._interpolate_arc_point(self._ps, idx_ps - 1, "30", self._ps[-1, 0], "TE_ps", 1.0005)

        # pressure-side projection
        self._interpolate_arc_point(self._ps_projected, idx_ps - 1, "31", self._ps_projected[-1, 0], "projection_ps")

        # block 7
        diff = (self._distance_projection[1] - self._distance_projection[0])
        self._all_nodes["38"] = deepcopy(self._all_nodes["31"])
        self._all_nodes["38"][0] += diff * self._chord
        self._all_nodes["38"][1] += diff * self._chord * _grad_ss     # intentional!
        self._all_nodes["39"] = deepcopy(self._all_nodes["32"])
        self._all_nodes["39"][0] += diff * self._chord
        self._all_nodes["39"][1] -= diff * self._chord * _grad_ss * 1.5     # intentional!
        self._create_back_nodes(40, 2)

        # add slight arc to left edge to improve the mesh TODO: rename all arcs for easier assignment
        y4 = (self._all_nodes["32"][1] + self._all_nodes["31"][1]) / 2
        x4 = self._distance_projection[0] * self._chord + self._ss[-1, 0] + self._chord
        self._all_arcs["TE_wake_block4"] = [x4*0.998, y4]

        y7 = (self._all_nodes["39"][1] + self._all_nodes["38"][1]) / 2
        x7 = (self._distance_projection[1]) * self._chord + self._ss[-1, 0] + self._chord
        self._all_arcs["TE_wake_block7"] = [x7*1.02, y7]

        # arcs for block 6 projection
        y6 = (self._all_nodes["26"][1] + self._all_nodes["39"][1]) / 2
        x6 = self._distance_projection[1] * self._chord + self._ss[-1, 0] + self._chord
        self._all_arcs["block6plus"] = [x6 * 0.89, y6 * 1.03]

        y6 = (self._all_nodes["28"][1] + self._all_nodes["38"][1]) / 2
        x6 = self._distance_projection[1] * self._chord + self._ps[0, 0] + self._chord
        self._all_arcs["block6minus"] = [x6 * 0.85, y6]

        # -------------------------------------- far field projection (2nd grid) --------------------------------------
        new_coord_ss = self._compute_projection(self._distance_projection[2] * self._chord, ss=True)
        new_coord_ps = self._compute_projection(self._distance_projection[2] * self._chord, ss=False)

        # block 8+
        self._all_nodes["42"] = new_coord_ss[-1, :].squeeze()
        self._all_nodes["43"] = deepcopy(self._all_nodes["20"])
        self._all_nodes["44"] = deepcopy(self._all_nodes["21"])
        self._all_nodes["45"] =  new_coord_ss[x_block_0, :].squeeze()

        for key in ["42", "45"]:
            self._all_nodes[key] = pt.cat([self._all_nodes[key], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._create_back_nodes(46, 4)

        # we need to duplicate the edge for merging
        self._all_edges["43_44"] = deepcopy(self._all_edges["20_21"])
        self._all_edges["42_45"] = new_coord_ss[x_block_0:, :]

        # block 8-
        self._all_nodes["50"] = new_coord_ps[x_block_0neg, :].squeeze()
        self._all_nodes["50"] = pt.cat([self._all_nodes["50"], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._all_nodes["51"] = deepcopy(self._all_nodes["24"])
        self._create_back_nodes(52, 2)
        self._all_edges["42_50"] = new_coord_ps[:x_block_0neg, :].squeeze()
        self._all_edges["43_51"] = deepcopy(self._all_edges["20_24"])

        # block 9+
        # interpolate projection SS airfoil
        inter = interp1d(new_coord_ss[:, 0], new_coord_ss[:, 1])
        y_coord_ss_pro3 = inter(self._rTE * new_coord_ss[0, 0])

        self._all_nodes["54"] = deepcopy(self._all_nodes["26"])
        self._all_nodes["55"] = pt.cat([self._rTE * new_coord_ss[0, 0].unsqueeze(-1),
                                        pt.from_numpy(y_coord_ss_pro3).unsqueeze(-1)], dim=-1)
        self._all_nodes["55"] = pt.cat([self._all_nodes["55"], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._create_back_nodes(56, 2)
        self._all_edges["44_54"] = deepcopy(self._all_edges["21_26"])
        self._all_edges["45_55"] = new_coord_ss[idx_ss:x_block_0, :]

        # block 9-
        # interpolate projection PS airfoil
        inter = interp1d(new_coord_ps[:, 0], new_coord_ps[:, 1])
        y_coord_ps_pro3 = inter(self._rTE * new_coord_ps[-1, 0])

        self._all_nodes["58"] = pt.cat([self._rTE * new_coord_ps[-1, 0].unsqueeze(-1),
                                        pt.from_numpy(y_coord_ps_pro3).unsqueeze(-1)], dim=-1)
        self._all_nodes["58"] = pt.cat([self._all_nodes["58"], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._all_nodes["59"] = deepcopy(self._all_nodes["28"])
        self._create_back_nodes(60, 2)

        # block 10+
        diff = (self._distance_projection[2] - self._distance_projection[1])
        self._all_nodes["62"] = deepcopy(self._all_nodes["39"])
        self._all_nodes["63"] = deepcopy(self._all_nodes["39"])
        self._all_nodes["63"][0] += diff * self._chord
        self._all_nodes["63"][1] -= diff * self._chord * _grad_ss * 1.5     # absicht!
        self._create_back_nodes(64, 2)
        self._all_arcs["block10plus"] = self._all_arcs["block6plus"] # no deepcopy here to ensure it's the same

        y10 = (self._all_nodes["55"][2] + self._all_nodes["63"][1]) / 2
        x10 = self._distance_projection[2] * self._chord + self._ss[-1, 0] + self._chord
        self._all_arcs["block10plusProj"] = [x10 * 1.02, y10 * 1.02]

        # block 10-
        self._all_nodes["66"] = deepcopy(self._all_nodes["38"])
        self._all_nodes["66"][0] += diff * self._chord
        self._all_nodes["66"][1] += diff * self._chord * _grad_ss    # absicht!
        self._all_nodes["67"] = deepcopy(self._all_nodes["38"])
        self._create_back_nodes(68, 2)
        self._all_arcs["block10minus"] = self._all_arcs["block6minus"] # no deepcopy here to ensure it's the same

        y10 = (self._all_nodes["58"][2] + self._all_nodes["66"][1]) / 2
        x10 = self._distance_projection[2] * self._chord + self._ps[0, 0] + self._chord
        self._all_arcs["block10minusProj"] = [x10 * 1.02, y10]

        # block 11
        self._all_arcs["block11"] = self._all_arcs["TE_wake_block7"]
        y11 = (self._all_nodes["63"][1] + self._all_nodes["66"][1]) / 2
        x11 = (self._distance_projection[2]) * self._chord + self._ss[-1, 0] + self._chord
        self._all_arcs["block11proj"] = [x11*1.02, y11]

        # -------------------------------------- far field projection (3rd grid)  --------------------------------------
        # same as in 2nd projection
        new_coord_ss = self._compute_projection(self._distance_projection[3] * self._chord, ss=True)
        new_coord_ps = self._compute_projection(self._distance_projection[3] * self._chord, ss=False)

        # block 12+
        self._all_nodes["70"] = new_coord_ss[-1, :].squeeze()
        self._all_nodes["71"] = deepcopy(self._all_nodes["42"])
        self._all_nodes["72"] = deepcopy(self._all_nodes["45"])
        self._all_nodes["73"] =  new_coord_ss[x_block_0, :].squeeze()

        for key in ["70", "73"]:
            self._all_nodes[key] = pt.cat([self._all_nodes[key], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._create_back_nodes(74, 4)

        # we need to duplicate the edge for merging
        self._all_edges["71_72"] = deepcopy(self._all_edges["42_45"])
        self._all_edges["70_73"] = new_coord_ss[x_block_0:, :]

        # block 12-
        self._all_nodes["78"] = new_coord_ps[x_block_0neg, :].squeeze()
        self._all_nodes["78"] = pt.cat([self._all_nodes["78"], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._all_nodes["79"] = deepcopy(self._all_nodes["50"])
        self._create_back_nodes(80, 2)
        self._all_edges["70_78"] = new_coord_ps[:x_block_0neg, :].squeeze()
        self._all_edges["71_79"] = deepcopy(self._all_edges["42_50"])

        # block 13+
        self._all_nodes["82"] = deepcopy(self._all_nodes["55"])
        self._all_nodes["82"] = pt.cat([self._all_nodes["82"], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._all_nodes["83"] = pt.cat([self._rTE * new_coord_ss[0, 0].unsqueeze(-1), new_coord_ss[0, 1].unsqueeze(-1),
                                        pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._create_back_nodes(84, 2)
        self._all_edges["72_82"] = deepcopy(self._all_edges["45_55"])
        self._all_edges["73_83"] = new_coord_ss[:x_block_0, :]

        # block 13-
        self._all_nodes["86"] = pt.cat([self._rTE * new_coord_ps[-1, 0].unsqueeze(-1),
                                       new_coord_ps[-1, 1].unsqueeze(-1), pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._all_nodes["87"] = deepcopy(self._all_nodes["58"])
        self._all_nodes["87"] = pt.cat([self._all_nodes["87"], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._create_back_nodes(88, 2)

        # block 14+
        diff = (self._distance_projection[3] - self._distance_projection[2])
        self._all_nodes["90"] = deepcopy(self._all_nodes["63"])
        self._all_nodes["91"] = deepcopy(self._all_nodes["63"])
        self._all_nodes["91"][0] += diff * self._chord
        self._all_nodes["91"][1] -= diff * self._chord * _grad_ss * 1.5     # absicht!
        self._create_back_nodes(92, 2)
        self._all_arcs["block14plus"] = self._all_arcs["block10plusProj"] # no deepcopy here to ensure it's the same

        y14 = (self._all_nodes["83"][2] + self._all_nodes["91"][1]) / 2
        x14 = self._distance_projection[3] * self._chord + self._ss[-1, 0] + self._chord
        self._all_arcs["block14plusProj"] = [x14 * 1.04, y14 * 1.04]

        # block 14-
        self._all_nodes["94"] = deepcopy(self._all_nodes["66"])
        self._all_nodes["95"] = deepcopy(self._all_nodes["66"])
        self._all_nodes["95"][0] += diff * self._chord
        self._all_nodes["95"][1] += diff * self._chord * _grad_ss     # absicht!
        self._create_back_nodes(96, 2)
        self._all_arcs["block14minus"] = self._all_arcs["block10minusProj"] # no deepcopy here to ensure it's the same

        # arc block 14 projection
        y14 = (self._all_nodes["86"][1] + self._all_nodes["95"][1]) / 2
        x14 = self._distance_projection[3] * self._chord + self._ps[0, 0] + self._chord
        self._all_arcs["block14minusProj"] = [x14*0.8, y14]         # at PS we always need correction factor due to asymmetry

        # block 15
        self._all_arcs["block15"] = self._all_arcs["block11proj"]

        y15 = (self._all_nodes["91"][1] + self._all_nodes["95"][1]) / 2
        x15 = (self._distance_projection[3]) * self._chord + self._ss[-1, 0] + self._chord
        self._all_arcs["block15proj"] = [x15*1.03, y15]

    def _interpolate_arc_point(self, curve: pt.Tensor, idx: int, te_node_key: str, x_ref: float, key: str,
                               scale_x: float = 1.0) -> None:
        """
        Interpolate and store an arc control point for blending the TE.

        :param curve: Curve coordinates
        :type curve: pt.Tensor
        :param idx: Index around which the interpolation stencil is constructed.
        :type idx: int
        :param te_node_key: Key into ``self._all_nodes`` identifying the trailing-edge node.
        :type te_node_key: str
        :param x_ref: Reference x-location used for trailing-edge blending.
        :type x_ref: float
        :param key: Key under which the arc point is stored in ``self._all_arcs``.
        :type key: str
        :param scale_x: Optional scaling factor applied to the interpolated x-location.
        :type scale_x: float
        :return: None
        :rtype: None
        """
        # TE node position
        pos_te = self._all_nodes[te_node_key][:2].unsqueeze(0)

        # build interpolation stencil based on central difference
        stencil = pt.cat([curve[idx - 1: idx + 2, :], pos_te], dim=0)
        interpolant = interp1d(stencil[:, 0], stencil[:, 1], kind="cubic")

        # compute blended x-location toward the trailing edge
        x_target = ((1.0 - self._rTE) / 1.5 + self._rTE) * x_ref
        y_target = interpolant(x_target)

        # store the interpolation point
        self._all_arcs[key] = [x_target * scale_x, y_target]

    def _create_back_nodes(self, start: int, n_nodes: int) -> None:
        """
        Create the nodes for the back patch of the mesh based on the front nodes.

        :param start: start index of the node
        :param n_nodes: number of back nodes to create
        :return: None
        :rtype: None
        """
        for i in range(start, start+n_nodes):
            self._all_nodes[str(i)] = deepcopy(self._all_nodes[str(i - n_nodes)])
            # overwrite the last dim -> back nodes (always use last dim, independent of orientation)
            self._all_nodes[str(i)][-1] = self._y_bound[0]

    def _write_header(self) -> None:
        """
        Write the header file of the blockMeshDict.

        :return: None
        :rtype: None
        """
        # write header and scale
        with open(join(self._write_path, self._block_mesh_name), "w") as f:
            f.writelines(self._header)
            f.writelines(self._scale)

    def _write_nodes(self, nodes: dict, completed: bool = False) -> None:
        """
        Write the nodes into the blockMeshDict.

        :return: None
        :rtype: None
        """
        # start the dict when we write the first node
        if list(nodes.keys())[0] == "0":
            lines = ["vertices\n(\n"]
        else:
            lines = []

        for n in nodes.keys():
            lines.append(f"\t( {nodes[n][0].item():{self._dec}} {nodes[n][-1].item():{self._dec}} "
                             f"{nodes[n][1].item():{self._dec}} ) \t\t//{n}\n")

        if completed:
            lines.append(");\n")

        with open(join(self._write_path, self._block_mesh_name), "a") as f:
            f.writelines(lines)

    def _write_blocks(self) -> None:
        """
        Write the blocks into the blockMeshDict.

        :return: None
        :rtype: None
        """
        # unpack the cells and gradings, format them -> n_cells are already ints
        ny = self._n_cells
        gy = [f"{g:{self._dec}}" for g in self._grading_normal]

        # multigrading for shock SS, PS and wake
        mg_ss = "((0.3 0.25 -2) (0.25 0.3 1) (0.3 0.325 1.2) (0.05 0.125 -6))"
        mg_ps = "((0.1 0.1 1.25) (0.8 0.7 1) (0.1 0.2 -6))"
        mg_w = "((0.5 0.5 1.2) (0.5 0.5 -1.2))"

        # keywords for grading and block type
        bt, eg, sg = "\n\thex", "edgeGrading", "simpleGrading"

        # number of cells and grading in spanwise direction (z-direction)
        nz, gz, egz = self._nz, 1, "1 1 1 1"

        # number of blocks in y-direction for the wake (3rd projection uses half of that value)
        ny_w = 20

        # gradings for block 5+ and 5-, 6+ and 6- (1st projection)
        gx5_6_plus = 1
        gx5_6_minus = -3

        # gradings for block 8+ and 8-, 10+ and 10- (2nd projection)
        gx8, gx10_plus = 3, 1

        # gradings for block 12+ and 12-, 14+ and 14- (3rd projection)
        gx12, gx14 = 3, 1

        # number of blocks in streamwise direction for the LE, SS and PS of the airfoil (only for the inner projection
        # ring)
        n_le, n_ss, n_ps = 45, 300, 180

        # we want the number of blocks in x-direction for blocks 8 and 12 to be the same (2nd/3rd projection),
        # but they are allowed to differ from the first projection
        nx_8_12 = 45

        # same for blocks 10 & 14 and 5 & 6
        nx_10_14, nx_5_6 = 30, 30

        # finally, the number of cells in streamwise direction for 9+ and 9- (blocks 13+ & 13- will have half of the
        # cells)
        nx9p, nx9m = 100, 70

        # create a list for the blocks
        lines = ["\nblocks\n("]

        blocks = [# block 0+    (0)
                 f"{bt} (0 1 2 3 4 5 6 7) ({ny[0]} {n_le} {nz}) {eg} (-{gy[0]} -{gy[0]} -{gy[0]} -{gy[0]} 5 8 8 5 {egz})",
                 # block 0-     (1)
                 f"{bt} (0 8 9 1 4 10 11 5) ({n_le} {ny[0]} {nz}) {eg} (6 15 15 6 -{gy[0]} -{gy[0]} -{gy[0]} -{gy[0]} {egz})",
                 # block 1+     (2)
                 f"{bt} (2 12 13 3 6 14 15 7) ({n_ss} {ny[0]} {nz}) {sg} ({mg_ss} {gy[0]} {gz})",
                 # block 1-     (3)
                 f"{bt} (8 16 17 9 10 18 19 11) ({n_ps} {ny[0]} {nz}) {sg} ({mg_ps} -{gy[0]} {gz})",
                 # block 2+     (4)
                 f"{bt} (20 0 3 21 22 4 7 23) ({ny[1]} {n_le} {nz}) {eg} (-{gy[1]} -{gy[1]} -{gy[1]} -{gy[1]} -3 5 5 -3 {egz})",
                 # block 2-     (5)
                 f"{bt} (20 24 8 0 22 25 10 4) ({n_le} {ny[1]} {nz}) {eg} (-3 6 6 -3 -{gy[1]} -{gy[1]} -{gy[1]} -{gy[1]} {egz})",
                 # block 3+     (6)
                 f"{bt} (3 13 26 21 7 15 27 23) ({n_ss} {ny[1]} {nz}) {sg} ({mg_ss} {gy[1]} {gz})",
                 # block 3-     (7)
                 f"{bt} (24 28 16 8 25 29 18 10) ({n_ps} {ny[1]} {nz}) {sg} ({mg_ps} -{gy[1]} {gz})",
                 # block 4      (8)
                 f"{bt} (30 31 32 33 34 35 36 37) ({ny[0]} {ny_w} {nz}) {sg} ({gy[0]} {mg_w} {gz})",
                 # block 5+     (9)
                 f"{bt} (12 33 32 13 14 37 36 15) ({nx_5_6} {ny[0]} {nz}) {sg} ({gx5_6_plus} {gy[0]} {gz})",
                 # block 5-     (10)
                 f"{bt} (16 31 30 17 18 35 34 19) ({nx_5_6} {ny[0]} {nz}) {sg} ({gx5_6_minus} -{gy[0]} {gz})",
                 # block 6+     (11)
                 f"{bt} (13 32 39 26 15 36 41 27) ({nx_5_6} {ny[1]} {nz}) {sg} ({gx5_6_plus} {gy[1]} {gz})",
                 # block 6-     (12)
                 f"{bt} (28 38 31 16 29 40 35 18) ({nx_5_6} {ny[1]} {nz}) {sg} ({gx5_6_minus} -{gy[1]} {gz})",
                 # block 7      (13)
                 f"{bt} (31 38 39 32 35 40 41 36) ({ny[1]} {ny_w} {nz}) {sg} ({gy[1]} {mg_w} {gz})",
                 # block 8+     (14)
                 f"{bt} (42 43 44 45 46 47 48 49) ({ny[2]} {nx_8_12} {nz}) {sg} (-{gy[2]} -{gx8} {gz})",
                 # block 8-     (15)
                 f"{bt} (42 50 51 43 46 52 53 47) ({nx_8_12} {ny[2]} {nz}) {sg} (-{gx8} -{gy[2]} {gz})",
                 # block 9+     (16)
                 f"{bt} (44 54 55 45 48 56 57 49) ({nx9p} {ny[2]} {nz}) {sg} ({mg_ss} {gy[2]} {gz})",
                 # block 9-     (17)
                 f"{bt} (50 58 59 51 52 60 61 53) ({nx9m} {ny[2]} {nz}) {sg} ({mg_ps} -{gy[2]} {gz})",
                 # block 10+    (18)
                 f"{bt} (54 62 63 55 56 64 65 57) ({nx_10_14} {ny[2]} {nz}) {sg} ({gx10_plus} {gy[2]} {gz})",
                 # block 10-    (19)
                 f"{bt} (58 66 67 59 60 68 69 61) ({nx_10_14} {ny[2]} {nz}) {sg} ({gx5_6_minus} -{gy[2]} {gz})",
                 # block 11     (20)
                 f"{bt} (67 66 63 62 69 68 65 64) ({ny[2]} {ny_w} {nz}) {sg} ({gy[2]} {mg_w} {gz})",
                 # block 12+    (21)
                 f"{bt} (70 71 72 73 74 75 76 77) ({ny[3]} {nx_8_12} {nz}) {sg} (-{gy[3]} -{gx12} {gz})",
                 # block 12-    (22)
                 f"{bt} (70 78 79 71 74 80 81 75) ({nx_8_12} {ny[3]} {nz}) {sg} (-{gx12} -{gy[3]} {gz})",
                 # block 13+    (23)
                 f"{bt} (72 82 83 73 76 84 85 77) ({nx9p//2} {ny[3]} {nz}) {sg} ({mg_ss} {gy[3]} {gz})",
                 # block 13-    (24)
                 f"{bt} (78 86 87 79 80 88 89 81) ({nx9m//2} {ny[3]} {nz}) {sg} ({mg_ps} -{gy[3]} {gz})",
                 # block 14+    (25)
                 f"{bt} (82 90 91 83 84 92 93 85) ({nx_10_14} {ny[3]} {nz}) {sg} ({gx14} {gy[3]} {gz})",
                 # block 14-    (26)
                 f"{bt} (86 95 94 87 88 97 96 89) ({nx_10_14} {ny[3]} {nz}) {sg} ({gx5_6_minus} -{gy[3]} {gz})",
                  # block 15    (27)
                 f"{bt} (90 94 95 91 92 96 97 93) ({ny_w} {ny[3]} {nz}) {sg} ({mg_w} {gy[3]} {gz})"]

        # add node numbers for easier debugging
        blocks = [l + f"\t// {i}" for i, l in enumerate(blocks)]

        #  close the node list
        lines += blocks
        lines.append( "\n);\n\n")

        with open(join(self._write_path, self._block_mesh_name), "a") as f:
            f.writelines(lines)

    def _write_edges(self, node_idx: List[Tuple], spline: pt.Tensor, completed: bool = False) -> None:
        """
        Write the edges into the blockMeshDict.

        :return: None
        :rtype: None
        """
        if self._init_edges:
            lines = []
        else:
            # before writing the first edge, we have to create a dict
            lines = ["edges\n("]
            self._init_edges = True

        # we need one spline for the front and one for the back of the domain
        for i in range(2):
            lines.append(f"\t\n\tspline {node_idx[i][0]} {node_idx[i][1]}\n\t(\n")
            if i == 0:
                for c in range(spline.size(0)):
                    lines.append(f"\t\t( {spline[c, 0].item():{self._dec}} {self._y_bound[1]:{self._dec}} "
                                 f"{spline[c, 1].item():{self._dec}} )\n")
            else:
                for c in range(spline.size(0)):
                    lines.append(f"\t\t( {spline[c, 0].item():{self._dec}} {self._y_bound[0]:{self._dec}} "
                                 f"{spline[c, 1].item():{self._dec}} )\n")
            lines.append("\t)\n")

        # once we have written the last edge, we have to clode the bracket in the edge dict of the blockMeshDict
        if completed:
            lines.append("\n);\n\n")

        with open(join(self._write_path, self._block_mesh_name), "a") as f:
            f.writelines(lines)

    def _write_arcs(self, node_idx: List[Tuple], point: Union[pt.Tensor, list], completed: bool = False) -> None:
        """
        Write the arcs into the blockMeshDict.

        :return: None
        :rtype: None
        """
        if self._init_edges:
            lines = []
        else:
            # before writing the first edge, we have to create a dict
            lines = ["edges\n("]
            self._init_edges = True

        # we need one spline for the front and one for the back of the domain
        for i in range(2):
            lines.append(f"\t\n\tarc {node_idx[i][0]} {node_idx[i][1]}\n\t(\n")
            if i == 0:
                lines.append(f"\t\t{point[0]:{self._dec}} {self._y_bound[1]:{self._dec}} {point[1]:{self._dec}}\n")
            else:
                lines.append(f"\t\t{point[0]:{self._dec}} {self._y_bound[0]:{self._dec}} {point[1]:{self._dec}}\n")
            lines.append("\t)\n")

        # once we have written the last edge, we have to clode the bracket in the edge dict of the blockMeshDict
        if completed:
            lines.append("\n);\n\n")

        with open(join(self._write_path, self._block_mesh_name), "a") as f:
            f.writelines(lines)

    @staticmethod
    def _assemble_boundaries(name: str, face_idx: list[Tuple], boundary_type: str) -> str:
        """
        Assemble the dicts for the boundary patches.

        :param name: name of the patch
        :param face_idx: face indices which make up the patch
        :param boundary_type: type of the boundary, e.g., patch
        :return: assembled boundary
        :rtype: str
        """
        _boundary_dict = f"\n\t{name}\n\t" + "{\n"
        _boundary_dict += f"\t\ttype {boundary_type};\n"
        _boundary_dict += "\t\tfaces\n\t\t(\n"
        for face in face_idx:
            _boundary_dict += f"\t\t\t( {" ".join([str(f) for f in face])} )\n"
        _boundary_dict += "\t\t);\n"
        _boundary_dict += "\t}\n"
        return _boundary_dict

    def _write_boundaries(self) -> None:
        """
        Write the boundaries into the blockMeshDict.

        :return:
        :rtype: None
        """
        lines = ["boundary\n("]

        # define front (inner mesh)
        _front = [(0, 1, 2, 3), (0, 8, 9, 1), (2, 12, 13, 3), (8, 9, 16, 17), (20, 0, 3, 21), (20, 24, 8, 0),
                  (3, 13, 26, 21), (24, 28, 16, 8), (30, 31, 32, 33), (33, 32, 13, 12), (16, 31, 30, 17),
                  (13, 32, 39, 26), (28, 38, 31, 16), (31, 38, 39, 32)]

        # define front (outer mesh)
        _front += [(42, 43, 44, 45), (44, 54, 55, 45), (54, 62, 63, 55), (62, 67, 66, 63), (58, 66, 67, 59),
                   (50, 58, 59, 51), (42, 50, 51, 43)]
        _front += [(70, 71, 72, 73), (72, 82, 83, 73), (82, 90, 91, 83), (90, 94, 95, 91), (86, 95, 94, 87),
                   (78, 86, 87, 79), (70, 78, 79, 71)]
        if self._nz == 1:
            lines.append(self._assemble_boundaries("front", _front, "empty"))
        else:
            lines.append(self._assemble_boundaries("front", _front, "cyclic;\n\t\tneighbourPatch  back"))

        # define back (inner mesh)
        _back = [(4, 5, 6, 7), (4, 10, 11, 5), (6, 14, 15, 7), (10, 18, 19, 11), (22, 4, 7, 23), (22, 25, 10, 4),
                 (7, 15, 27, 23), (25, 29, 18, 10), (34, 35, 36, 37), (14, 37, 36, 15), (18, 35, 34, 19),
                 (15, 36, 41, 27), (29, 40, 35, 18), (35, 40, 41, 36)]

        # define back (outer mesh)
        _back += [(46, 47, 48, 49), (48, 56, 57, 49), (56, 64, 65, 57), (64, 69, 68, 65), (60, 68, 69, 61),
                  (52, 60, 61, 53), (46, 52, 53, 47)]
        _back += [(74, 75, 76, 77), (76, 84, 85, 77), (84, 92, 93, 85), (92, 96, 97, 93), (88, 97, 96, 89),
                  (80, 88, 89, 81), (74, 80, 81, 75)]
        if self._nz == 1:
            lines.append(self._assemble_boundaries("back", _back, "empty"))
        else:
            lines.append(self._assemble_boundaries("back", _back, "cyclic;\n\t\tneighbourPatch  front"))

        # define airfoil
        _airfoil = [(1, 2, 6, 5), (2, 12, 14, 6), (12, 33, 37, 14), (33, 37, 34, 30), (30, 17, 19, 34),
                    (17, 9, 11, 19), (9, 1, 5, 11)]
        lines.append(self._assemble_boundaries("airfoil", _airfoil, "wall"))

        # define mesh boundary to the far field mesh (inner mesh)
        _inner_bound = [(20, 21, 23, 22), (21, 26, 27, 23), (26, 39, 41, 27), (39, 38, 40, 41), (38, 28, 29, 40),
                        (28, 24, 25, 29), (24, 20, 22, 25)]
        lines.append(self._assemble_boundaries("inner_boundary", _inner_bound, "patch"))

        # define mesh boundary to the far field mesh (outer mesh)
        _inner_bound_proj = [(43, 44, 48, 47), (44, 54, 56, 48), (54, 62, 64, 56), (62, 67, 69, 64), (67, 59, 61, 69),
                             (51, 59, 61, 53), (43, 51, 53, 47)]
        lines.append(self._assemble_boundaries("inner_boundary_projection", _inner_bound_proj, "patch"))

        # define inner mesh boundary between the 2nd and 3rd projection (2nd far field mesh)
        _inner_bound_proj2 = [(42, 45, 49, 46), (45, 55, 57, 49), (55, 63, 65, 57), (63, 66, 68, 65), (66, 58, 60, 68),
                         (50, 58, 60, 52), (42, 50, 52, 46)]
        lines.append(self._assemble_boundaries("inner_boundary_projection2", _inner_bound_proj2, "patch"))

        # define outer mesh boundary between the 2nd and 3rd projection (2nd far field mesh)
        _inner_bound_proj3 = [(71, 75, 76, 72), (72, 76, 84, 82), (82, 84, 92, 90), (90, 92, 96, 94), (94, 96, 89, 87),
                              (87, 89, 81, 79), (79, 81, 75, 71)]
        lines.append(self._assemble_boundaries("inner_boundary_projection3", _inner_bound_proj3, "patch"))

        _inlet_outlet = [(70, 74, 77, 73), (73, 77, 85, 83), (83, 85, 93, 91), (91, 93, 97, 95), (95, 97, 88, 86),
                         (86, 88, 80, 78), (78, 80, 74, 70)]
        lines.append(self._assemble_boundaries("inlet_outlet", _inlet_outlet, "patch"))
        lines.append("\n);\n\n")
        with open(join(self._write_path, self._block_mesh_name), "a") as f:
            f.writelines(lines)

    def _write_merge_patch_pairs(self) -> None:
        """
        Write the faces to merge into the blockMeshDict.

        :return:
        :rtype: None
        """
        lines = ["\nmergePatchPairs\n(\n"
                 "\t(inner_boundary inner_boundary_projection)\n"
                 "\t(inner_boundary_projection2 inner_boundary_projection3)\n"
                 ");\n\n"]
        with open(join(self._write_path, self._block_mesh_name), "a") as f:
            f.writelines(lines)

    def _write_footer(self) -> None:
        """
        Write the footer of the blockMeshDict.

        :return:
        :rtype: None
        """
        with open(join(self._write_path, self._block_mesh_name), "a") as f:
            f.writelines(self._footer)


if __name__ == "__main__":
    pass
