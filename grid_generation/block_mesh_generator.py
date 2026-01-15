"""
class for generating a blockMesh based on coordinates
"""
from copy import deepcopy
from os.path import join
from typing import Union, Tuple, List
import torch as pt
import numpy as np
from pandas import read_csv
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
                 y_min: Union[int, float] = -0.0375, y_max: Union[int, float]=0,
                 distance_LE: Union[int, float] = 0.05, first_layer_thickness: Union[int, float]=1e-5,
                 use_yPlus: bool = False, expansion_rate: Union[int, float] = 1.25, projection_distance: list = None,
                 reynolds_number: Union[int, float] = None, yPlus_target: float = 1, u_infinity: Union[int, float] = 1,
                 density: Union[int, float] = 1):

        # set default distances
        if projection_distance is None:
            projection_distance = [0.01, 1.5, 2.5, 50]

        assert len(projection_distance) == 4, "TODO"

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

        # airfoil properties
        self._ss = None
        self._ps = None
        self._chord = None

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

    def generate_block_mesh(self):
        self._read_coordinates()
        self._compute_n_blocks_and_grading_normal()
        self._project_coordinates()
        self._assemble_nodes()

    def write_block_mesh_dict(self):
        self._write_header()
        self._write_nodes(self._all_nodes, completed=True)
        self._write_blocks()

        # write arcs for TE (SS)
        self._write_arcs([(33, 12), (37, 14)], self._all_arcs["TE_ss"])
        self._write_arcs([(13, 32), (15, 36)], self._all_arcs["projection_ss"])

        # write arcs for TE (PS)
        self._write_arcs([(30, 17), (34, 19)], self._all_arcs["TE_ps"])
        self._write_arcs([(31, 16), (35, 18)], self._all_arcs["projection_ps"])

        # write arcs for TE
        # self._write_arcs([(33, 30), (34, 37)], self._all_arcs["TE"])

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
        # don't use this edge
        # self._write_edges([(24, 28), (25, 29)], self._all_edges["25_29"])

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

    def _read_coordinates(self):
        _coordinates = pt.from_numpy(read_csv(join(self._load_path, self._file_name), sep=r"\s+", skiprows=1,
                                              header=None, names=["x", "y"]).to_numpy())
        self._chord = _coordinates[:, 0].max().item() - _coordinates[:, 0].min().item()

        # split into SS and PS
        LE_idx = pt.where(_coordinates[:, 0] == _coordinates[:, 0].min())[0]
        self._ss = _coordinates[:LE_idx+1]
        self._ps =  _coordinates[LE_idx:]

    def _compute_n_blocks_and_grading_normal(self):
        for i in range(len(self._distance_projection)):
            if i == 0:
                if self._use_yPlus:
                    # alternatively compute number of cells and grading for a given inflow condition and yPlus
                    cf = 0.026 * self._re ** (-1.0 / 7.0)
                    tau_wall = 0.5 * cf * self._density * self._u_infinity ** 2
                    u_tau = np.sqrt(tau_wall / self._density)
                    mu = self._density * self._chord * self._u_infinity / self._re
                    first_layer_thk = mu * self._yPlus_target / (self._density * u_tau)
                else:
                    first_layer_thk = self._first_layer_thickness[0]
                _dist = self._distance_projection[i] * self._chord
            else:
                first_layer_thk = self._last_cell_thickness[-1]
                _dist = (self._distance_projection[i] - self._distance_projection[i-1]) * self._chord

            # TODO: documentation!
            _e = pt.tensor(self._expansion_rate[i])
            if abs(_e - 1.0) < 1e-12:
                # uniform spacing for e = 1
                n_cells = pt.ceil(pt.tensor(_dist / first_layer_thk))
                _thickness_new = n_cells * first_layer_thk
                _grading = pt.ones(1,)
            else:
                n_cells = pt.ceil(pt.log(pt.ones(1,) + (_e - pt.ones(1,)) * _dist / first_layer_thk) / pt.log(_e))
                _thickness_new = first_layer_thk * (_e **n_cells - 1.0) / (_e - 1.0)
                _grading = _e**(n_cells - 1)

            # store the computed cells and gradings
            self._grading_normal.append(_grading.item())
            self._n_cells.append(int(n_cells.item()))
            if i == 0:
                self._distance_projection[i] = _thickness_new.item() / self._chord
            else:
                self._distance_projection[i] = (_thickness_new.item() + self._distance_projection[i-1]) * self._chord
            self._last_cell_thickness.append((first_layer_thk * _e**(n_cells - 1)).item())

    def _project_coordinates(self):
        self._get_normal_vectors()
        self._ss_projected = self._compute_projection(self._distance_projection[0] * self._chord, True)
        self._ps_projected = self._compute_projection(self._distance_projection[0] * self._chord, False)

    @staticmethod
    def _compute_normals(coordinates):
        # compute tangent vectors
        diffs = np.gradient(coordinates, axis=0)
        dx, dy = diffs[:, 0], diffs[:, 1]

        # normalize tangent vectors
        tangent_norm = np.sqrt(dx**2 + dy**2)
        tx, ty = dx / tangent_norm, dy / tangent_norm

        # compute normal vectors
        nx, ny = ty, -tx

        return nx, ny

    def _get_normal_vectors(self):
        self._n_ss = self._compute_normals(self._ss)
        self._n_ps = self._compute_normals(self._ps)

    def _compute_projection(self, offset, ss):
        if ss:
            return pt.stack([self._ss[:, 0] + offset * self._n_ss[0], self._ss[:, 1] + offset * self._n_ss[1]]).transpose(-1, 0)
        else:
            return pt.stack([self._ps[:, 0] + offset * self._n_ps[0], self._ps[:, 1] + offset * self._n_ps[1]]).transpose(-1, 0)

    def _assemble_nodes(self):
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

        # block 1+, shift nodes at TE to 0.XXc (manually) to create round TE
        round_TE = 0.998

        # interpolate SS airfoil TE
        inter = interp1d(self._ss[:, 0], self._ss[:, 1])
        y_coord_ss = inter(round_TE * self._ss[0, 0])

        # interpolate projection SS airfoil TE
        inter = interp1d(self._ss_projected[:, 0], self._ss_projected[:, 1])
        y_coord_ss_pro = inter(round_TE * self._ss_projected[0, 0])

        self._all_nodes["12"] = pt.cat([round_TE * self._ss[0, 0].unsqueeze(-1), pt.from_numpy(y_coord_ss).unsqueeze(-1)],
                                       dim=-1)
        self._all_nodes["13"] = pt.cat([round_TE * self._ss_projected[0, 0].unsqueeze(-1),
                                        pt.from_numpy(y_coord_ss_pro).unsqueeze(-1)], dim=-1)

        # add 3rd dim for front
        for key in ["12", "13"]:
            self._all_nodes[key] = pt.cat([self._all_nodes[key], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._create_back_nodes(14, 2)

        # crop airfoil coordinates based on y_coord_ss & y_coord_ss_proj
        idx_ss = pt.where(self._ss[:, 0] < round_TE * self._ss[0, 0])[0][0].item()
        self._all_edges["2_12"] = self._ss[idx_ss:x_block_0, :]
        self._all_edges["13_3"] = self._ss_projected[idx_ss:x_block_0, :]

        # block 1-
        # interpolate PS airfoil TE
        inter = interp1d(self._ps[:, 0], self._ps[:, 1])
        y_coord_ps = inter(round_TE * self._ps[-1, 0])

        # interpolate projection SS airfoil TE
        inter = interp1d(self._ps_projected[:, 0], self._ps_projected[:, 1])
        y_coord_ps_pro = inter(round_TE * self._ps_projected[-1, 0])

        self._all_nodes["16"] = pt.cat([round_TE * self._ps_projected[-1, 0].unsqueeze(-1),
                                        pt.from_numpy(y_coord_ps_pro).unsqueeze(-1)], dim=-1)
        self._all_nodes["17"] = pt.cat([round_TE * self._ps[-1, 0].unsqueeze(-1),
                                        pt.from_numpy(y_coord_ps).unsqueeze(-1)], dim=-1)
        for key in ["16", "17"]:
            self._all_nodes[key] = pt.cat([self._all_nodes[key], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._create_back_nodes(18, 2)

        # crop airfoil coordinates based on y_coord_ss & y_coord_ss_proj
        idx_ps = pt.where(self._ps[:, 0] < round_TE * self._ps[-1, 0])[0][-1].item()

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
        y_coord_ss_pro2 = inter(round_TE * new_coord_ss[0, 0])

        # interpolate projection PS airfoil
        inter = interp1d(new_coord_ps[:, 0], new_coord_ps[:, 1])
        y_coord_ps_pro2 = inter(round_TE * new_coord_ps[-1, 0])

        self._all_nodes["26"] = pt.cat([round_TE * new_coord_ss[0, 0].unsqueeze(-1),
                                        pt.from_numpy(y_coord_ss_pro2).unsqueeze(-1)], dim=-1)
        for key in ["26"]:
            self._all_nodes[key] = pt.cat([self._all_nodes[key], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._create_back_nodes(27, 1)
        self._all_edges["21_26"] = new_coord_ss[idx_ss:x_block_0, :]

        # block 3-
        self._all_nodes["28"] = pt.cat([round_TE * new_coord_ps[-1, 0].unsqueeze(-1),
                                        pt.from_numpy(y_coord_ps_pro2).unsqueeze(-1)], dim=-1)

        for key in ["28"]:
            self._all_nodes[key] = pt.cat([self._all_nodes[key], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._create_back_nodes(29, 1)

        # now get the gradient of the last point to continue the block with the same curvature, orient on SS since more
        # important
        _grad_ss = (self._ss[0, 1] - self._ss[1, 1]) / (self._ss[0, 0] - self._ss[1, 0])
        _grad_ps = (self._ps[-1, 1] - self._ps[-2, 1]) / (self._ps[-1, 0] - self._ps[-2, 0])

        # block 4 at TE
        # move the nodes by the same distance in y at TE as we did on the airfoil's surface
        y_pos_TE_PS = self._ps[-1, 1] + (1-round_TE) * self._chord
        y_pos_TE_SS = self._ss[0, 1] - (1-round_TE) * self._chord

        # TODO: add assertion in case round_TE leads to negative block 4
        if y_pos_TE_SS < y_pos_TE_PS:
            raise ArithmeticError

        self._all_nodes["30"] = pt.cat([self._ps[-1, 0].unsqueeze(-1), y_pos_TE_PS.unsqueeze(-1),
                                        pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._all_nodes["31"] = deepcopy(self._all_nodes["17"])
        self._all_nodes["31"][0] += self._distance_projection[0] * self._chord
        self._all_nodes["31"][1] += self._distance_projection[0] * self._chord * _grad_ss     # absicht!
        self._all_nodes["32"] = deepcopy(self._all_nodes["12"])
        self._all_nodes["32"][0] += self._distance_projection[0] * self._chord
        self._all_nodes["32"][1] -= self._distance_projection[0] * self._chord * _grad_ss     # absicht!

        # block 5+
        self._all_nodes["33"] = pt.cat([self._ss[0, 0].unsqueeze(-1), y_pos_TE_SS.unsqueeze(-1),
                                        pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._create_back_nodes(34, 4)

        # compute point for arc at TE
        _pos_TE_SS = self._all_nodes["33"][:2].unsqueeze(0)
        _test = pt.cat([self._ss[idx_ss-1:idx_ss + 2, :], _pos_TE_SS], dim=0)
        inter = interp1d(_test[:, 0], _test[:, 1], kind="cubic")
        y_inter = inter(((1 - round_TE)/1.5 + round_TE) * self._ss[0, 0])
        self._all_arcs["TE_ss"] = [((1 - round_TE)/1.5 + round_TE) * self._ss[0, 0], y_inter]

        # apply same procedure for remaining arcs
        _pos_TE_SS = self._all_nodes["32"][:2].unsqueeze(0)
        _test = pt.cat([self._ss_projected[idx_ss-1:idx_ss + 2, :], _pos_TE_SS], dim=0)
        inter = interp1d(_test[:, 0], _test[:, 1], kind="cubic")
        y_inter = inter(((1 - round_TE)/1.5 + round_TE) * self._ss_projected[0, 0])
        self._all_arcs["projection_ss"] = [((1 - round_TE)/1.5 + round_TE) * self._ss_projected[0, 0], y_inter]

        # block 5-
        _pos_TE_PS = self._all_nodes["30"][:2].unsqueeze(0)
        _test = pt.cat([self._ps[idx_ps-2:idx_ps + 1, :], _pos_TE_PS], dim=0)
        inter = interp1d(_test[:, 0], _test[:, 1], kind="cubic")
        y_inter = inter(((1 - round_TE)/1.5 + round_TE) * self._ps[-1, 0])
        self._all_arcs["TE_ps"] = [((1 - round_TE)/1.5 + round_TE) * self._ps[-1, 0] * 1.0005, y_inter]

        _pos_TE_PS = self._all_nodes["31"][:2].unsqueeze(0)
        _test = pt.cat([self._ps_projected[idx_ps-2:idx_ps + 1, :], _pos_TE_PS], dim=0)
        inter = interp1d(_test[:, 0], _test[:, 1], kind="cubic")
        y_inter = inter(((1 - round_TE)/1.5 + round_TE) * self._ps_projected[-1, 0])
        self._all_arcs["projection_ps"] = [((1 - round_TE)/1.5 + round_TE) * self._ps_projected[-1, 0], y_inter]

        # TE arc
        # self._all_arcs["TE"] = [self._ss[0, 0] + 2e-5, self._ps[-1, 1] + (self._ss[0, 1] - self._ps[-1, 1]).abs() / 2]

        # block 7
        diff = (self._distance_projection[1] - self._distance_projection[0])
        self._all_nodes["38"] = deepcopy(self._all_nodes["31"])
        self._all_nodes["38"][0] += diff * self._chord
        self._all_nodes["38"][1] += diff * self._chord * _grad_ss     # absicht!
        self._all_nodes["39"] = deepcopy(self._all_nodes["32"])
        self._all_nodes["39"][0] += diff * self._chord
        self._all_nodes["39"][1] -= diff * self._chord * _grad_ss     # absicht!
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
        self._all_arcs["block6plus"] = [x6 * 0.88, y6]

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
        y_coord_ss_pro3 = inter(round_TE * new_coord_ss[0, 0])

        self._all_nodes["54"] = deepcopy(self._all_nodes["26"])
        self._all_nodes["55"] = pt.cat([round_TE * new_coord_ss[0, 0].unsqueeze(-1),
                                        pt.from_numpy(y_coord_ss_pro3).unsqueeze(-1)], dim=-1)
        self._all_nodes["55"] = pt.cat([self._all_nodes["55"], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._create_back_nodes(56, 2)
        self._all_edges["44_54"] = deepcopy(self._all_edges["21_26"])
        self._all_edges["45_55"] = new_coord_ss[idx_ss:x_block_0, :]

        # block 9-
        # interpolate projection PS airfoil
        inter = interp1d(new_coord_ps[:, 0], new_coord_ps[:, 1])
        y_coord_ps_pro3 = inter(round_TE * new_coord_ps[-1, 0])

        self._all_nodes["58"] = pt.cat([round_TE * new_coord_ps[-1, 0].unsqueeze(-1),
                                        pt.from_numpy(y_coord_ps_pro3).unsqueeze(-1)], dim=-1)
        self._all_nodes["58"] = pt.cat([self._all_nodes["58"], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._all_nodes["59"] = deepcopy(self._all_nodes["28"])
        self._create_back_nodes(60, 2)

        # block 10+
        diff = (self._distance_projection[2] - self._distance_projection[1])
        self._all_nodes["62"] = deepcopy(self._all_nodes["39"])
        self._all_nodes["63"] = deepcopy(self._all_nodes["39"])
        self._all_nodes["63"][0] += diff * self._chord
        self._all_nodes["63"][1] -= diff * self._chord * _grad_ss     # absicht!
        self._create_back_nodes(64, 2)
        self._all_arcs["block10plus"] = self._all_arcs["block6plus"] # no deepcopy here to ensure it's the same

        y10 = (self._all_nodes["55"][2] + self._all_nodes["63"][1]) / 2
        x10 = self._distance_projection[2] * self._chord + self._ss[-1, 0] + self._chord
        self._all_arcs["block10plusProj"] = [x10 * 1.02, y10]

        # block 10-
        self._all_nodes["66"] = deepcopy(self._all_nodes["38"])
        self._all_nodes["66"][0] += diff * self._chord
        self._all_nodes["66"][1] += diff * self._chord * _grad_ss     # absicht!
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
        self._all_nodes["83"] = pt.cat([round_TE * new_coord_ss[0, 0].unsqueeze(-1), new_coord_ss[0, 1].unsqueeze(-1),
                                        pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._create_back_nodes(84, 2)
        self._all_edges["72_82"] = deepcopy(self._all_edges["45_55"])
        self._all_edges["73_83"] = new_coord_ss[:x_block_0, :]

        # block 13-
        self._all_nodes["86"] = pt.cat([round_TE * new_coord_ps[-1, 0].unsqueeze(-1),
                                       new_coord_ps[-1, 1].unsqueeze(-1), pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._all_nodes["87"] = deepcopy(self._all_nodes["58"])
        self._all_nodes["87"] = pt.cat([self._all_nodes["87"], pt.tensor(self._y_bound[1]).unsqueeze(-1)], dim=-1)
        self._create_back_nodes(88, 2)

        # block 14+
        diff = (self._distance_projection[3] - self._distance_projection[2])
        self._all_nodes["90"] = deepcopy(self._all_nodes["63"])
        self._all_nodes["91"] = deepcopy(self._all_nodes["63"])
        self._all_nodes["91"][0] += diff * self._chord
        self._all_nodes["91"][1] -= diff * self._chord * _grad_ss     # absicht!
        self._create_back_nodes(92, 2)
        self._all_arcs["block14plus"] = self._all_arcs["block10plusProj"] # no deepcopy here to ensure it's the same

        y14 = (self._all_nodes["83"][2] + self._all_nodes["91"][1]) / 2
        x14 = self._distance_projection[3] * self._chord + self._ss[-1, 0] + self._chord
        self._all_arcs["block14plusProj"] = [x14 * 1.02, y14]

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
        self._all_arcs["block15proj"] = [x15*1.02, y15]

    def _create_back_nodes(self, start: int, n_nodes: int):
        for i in range(start, start+n_nodes):
            self._all_nodes[str(i)] = deepcopy(self._all_nodes[str(i - n_nodes)])
            # overwrite the last dim -> back nodes (always use last dim, independent of orientation)
            self._all_nodes[str(i)][-1] = self._y_bound[0]

    def _write_header(self):
        # write header and scale
        with open(join(self._write_path, self._block_mesh_name), "w") as f:
            f.writelines(self._header)
            f.writelines(self._scale)

    def _write_nodes(self, nodes: dict, completed: bool = False):
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
        # unpack the cells and gradings, format them -> n_cells are already ints,
        ny = self._n_cells
        gy = [f"{g:{self._dec}}" for g in self._grading_normal]

        # multigrading for shock SS
        mg_ss = "((0.3 0.25 -2) (0.25 0.3 1) (0.3 0.325 1.2) (0.05 0.125 -5))"

        # write the blocks
        lines = ["\nblocks\n(",
                 # block 0+
                 f"\n\thex (0 1 2 3 4 5 6 7) ({ny[0]} 45 1) edgeGrading (-{gy[0]} -{gy[0]} -{gy[0]} -{gy[0]} 5 8 8 5 1 1 1 1)  // 0",
                 # block 0-
                 f"\n\thex (0 8 9 1 4 10 11 5) (45 {ny[0]} 1) edgeGrading (6 15 15 6 -{gy[0]} -{gy[0]} -{gy[0]} -{gy[0]} 1 1 1 1)  // 1",
                 # block 1+
                 f"\n\thex (2 12 13 3 6 14 15 7) (300 {ny[0]} 1) simpleGrading ({mg_ss} {gy[0]} 1)  // 2",
                 # block 1-
                 f"\n\thex (8 16 17 9 10 18 19 11) (180 {ny[0]} 1) simpleGrading (((0.1 0.1 1.25) (0.8 0.7 1) (0.1 0.2 -4)) -{gy[0]} 1)  // 3",
                 # block 2+
                 f"\n\thex (20 0 3 21 22 4 7 23) ({ny[1]} 45 1) edgeGrading (-{gy[1]} -{gy[1]} -{gy[1]} -{gy[1]} -3 5 5 -3 1 1 1 1)  // 4",
                 # block 2-
                 f"\n\thex (20 24 8 0 22 25 10 4) (45 {ny[1]} 1) edgeGrading (-3 6 6 -3 -{gy[1]} -{gy[1]} -{gy[1]} -{gy[1]} 1 1 1 1)  // 5",
                 # block 3+
                 f"\n\thex (3 13 26 21 7 15 27 23) (300 {ny[1]} 1) simpleGrading ({mg_ss} {gy[1]} 1)  // 6",
                 # block 3-
                 f"\n\thex (24 28 16 8 25 29 18 10) (180 {ny[1]} 1) simpleGrading (((0.1 0.1 1.25) (0.8 0.7 1) (0.1 0.2 -4)) -{gy[1]} 1)  // 7",
                 # block 4
                 f"\n\thex (30 31 32 33 34 35 36 37) ({ny[0]} 20 1) simpleGrading ({gy[0]} ((0.5 0.5 1.2) (0.5 0.5 -1.2)) 1)  // 8",
                 # block 5+
                 f"\n\thex (12 33 32 13 14 37 36 15) (30 {ny[0]} 1) simpleGrading (1 {gy[0]} 1)  // 9",
                 # block 5-
                 f"\n\thex (16 31 30 17 18 35 34 19) (30 {ny[0]} 1) simpleGrading (1 -{gy[0]} 1)  // 10",
                 # block 6+
                 f"\n\thex (13 32 39 26 15 36 41 27) (30 {ny[1]} 1) simpleGrading (1 {gy[1]} 1)  // 11",
                 # block 6-
                 f"\n\thex (28 38 31 16 29 40 35 18) (30 {ny[1]} 1) simpleGrading (1 -{gy[1]} 1)  // 12",
                 # block 7
                 f"\n\thex (31 38 39 32 35 40 41 36) ({ny[1]} 20 1) simpleGrading ({gy[1]} ((0.5 0.5 1.2) (0.5 0.5 -1.2)) 1)  // 13",
                 # block 8+
                 f"\n\thex (42 43 44 45 46 47 48 49) ({ny[2]} 45 1) simpleGrading (-{gy[2]} -3 1)  // 14",
                 # block 8-
                 f"\n\thex (42 50 51 43 46 52 53 47) (45 {ny[2]} 1) simpleGrading (-3 -{gy[2]} 1)  // 15",
                 # block 9+
                 f"\n\thex (44 54 55 45 48 56 57 49) (100 {ny[2]} 1) simpleGrading ({mg_ss} {gy[2]} 1)  // 16",
                 # block 9-
                 f"\n\thex (50 58 59 51 52 60 61 53) (70 {ny[2]} 1) simpleGrading (((0.1 0.1 1.25) (0.8 0.7 1) (0.1 0.2 -4)) -{gy[2]} 1)  // 17",
                 # block 10+
                 f"\n\thex (54 62 63 55 56 64 65 57) (30 {ny[2]} 1) simpleGrading (1 {gy[2]} 1)  // 18",
                 # block 10-
                 f"\n\thex (58 66 67 59 60 68 69 61) (30 {ny[2]} 1) simpleGrading (1 -{gy[2]} 1)  // 19",
                 # block 11
                 f"\n\thex (67 66 63 62 69 68 65 64) ({ny[2]} 20 1) simpleGrading ({gy[2]} ((0.5 0.5 1.2) (0.5 0.5 -1.2)) 1)  // 20",
                 # block 12+
                 f"\n\thex (70 71 72 73 74 75 76 77) ({ny[3]} 45 1) simpleGrading (-{gy[3]} -3 1)  // 21",
                 # block 12-
                 f"\n\thex (70 78 79 71 74 80 81 75) (45 {ny[3]} 1) simpleGrading (-3 -{gy[3]} 1)  // 22",
                 # block 13+
                 f"\n\thex (72 82 83 73 76 84 85 77) (50 {ny[3]} 1) simpleGrading ({mg_ss} {gy[3]} 1)  // 23",
                 # block 13-
                 f"\n\thex (78 86 87 79 80 88 89 81) (35 {ny[3]} 1) simpleGrading (((0.1 0.1 1.25) (0.8 0.7 1) (0.1 0.2 -4)) -{gy[3]} 1)  // 24",
                 # block 14+
                 f"\n\thex (82 90 91 83 84 92 93 85) (30 {ny[3]} 1) simpleGrading (1 {gy[3]} 1)  // 25",
                 # block 14-
                 f"\n\thex (86 95 94 87 88 97 96 89) (30 {ny[3]} 1) simpleGrading (1 -{gy[3]} 1)  // 26",
                  # block 15
                 f"\n\thex (90 94 95 91 92 96 97 93) (10 {ny[3]} 1) simpleGrading (((0.5 0.5 1.2) (0.5 0.5 -1.2)) {gy[3]} 1)  // 27",
                 "\n);\n\n"]
        # TODO: loop over list, add node number to str, then close the node list

        with open(join(self._write_path, self._block_mesh_name), "a") as f:
            f.writelines(lines)

    def _write_edges(self, node_idx: List[Tuple], spline: pt.Tensor, completed: bool = False):
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
        _boundary_dict = f"\n\t{name}\n\t" + "{\n"
        _boundary_dict += f"\t\ttype {boundary_type};\n"
        _boundary_dict += "\t\tfaces\n\t\t(\n"
        for face in face_idx:
            _boundary_dict += f"\t\t\t( {" ".join([str(f) for f in face])} )\n"
        _boundary_dict += "\t\t);\n"
        _boundary_dict += "\t}\n"
        return _boundary_dict

    def _write_boundaries(self) -> None:
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
        lines.append(self._assemble_boundaries("front", _front, "empty"))

        # define back (inner mesh)
        _back = [(4, 5, 6, 7), (4, 10, 11, 5), (6, 14, 15, 7), (10, 18, 19, 11), (22, 4, 7, 23), (22, 25, 10, 4),
                 (7, 15, 27, 23), (25, 29, 18, 10), (34, 35, 36, 37), (14, 37, 36, 15), (18, 35, 34, 19),
                 (15, 36, 41, 27), (29, 40, 35, 18), (35, 40, 41, 36)]

        # define back (outer mesh)
        _back += [(46, 47, 48, 49), (48, 56, 57, 49), (56, 64, 65, 57), (64, 69, 68, 65), (60, 68, 69, 61),
                  (52, 60, 61, 53), (46, 52, 53, 47)]
        _back += [(74, 75, 76, 77), (76, 84, 85, 77), (84, 92, 93, 85), (92, 96, 97, 93), (88, 97, 96, 89),
                  (80, 88, 89, 81), (74, 80, 81, 75)]
        lines.append(self._assemble_boundaries("back", _back, "empty"))

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

        # define inlet & outlet as single 'inlet_outlet' patch
        """_inlet = [(70, 74, 77, 73), (73, 77, 85, 83), (86, 88, 80, 78), (78, 80, 74, 70)]
        lines.append(self._assemble_boundaries("inlet", _inlet, "patch"))
        _outlet = [(83, 85, 93, 91), (91, 93, 97, 95), (95, 97, 88, 86)]
        lines.append(self._assemble_boundaries("outlet", _outlet, "patch"))"""

        _inlet_outlet = [(70, 74, 77, 73), (73, 77, 85, 83), (83, 85, 93, 91), (91, 93, 97, 95), (95, 97, 88, 86),
                         (86, 88, 80, 78), (78, 80, 74, 70)]
        lines.append(self._assemble_boundaries("inlet_outlet", _inlet_outlet, "patch"))
        lines.append("\n);\n\n")
        with open(join(self._write_path, self._block_mesh_name), "a") as f:
            f.writelines(lines)

    def _write_merge_patch_pairs(self) -> None:
        lines = ["\nmergePatchPairs\n(\n"
                 "\t(inner_boundary inner_boundary_projection)\n"
                 "\t(inner_boundary_projection2 inner_boundary_projection3)\n"
                 ");\n\n"]
        with open(join(self._write_path, self._block_mesh_name), "a") as f:
            f.writelines(lines)

    def _write_footer(self) -> None:
        with open(join(self._write_path, self._block_mesh_name), "a") as f:
            f.writelines(self._footer)


if __name__ == "__main__":
    pass
