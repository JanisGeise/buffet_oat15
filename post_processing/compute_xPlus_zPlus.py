"""
compute z+ and x+ based on surface data for the DDES setup in order to ensure a good grid quality
"""
from os import makedirs

import torch as pt
import matplotlib.pyplot as plt

from os.path import join, exists
from flowtorch.data import CSVDataloader
from flowtorch.data.utils import CellWeightEstimator

from utils import compute_camber_line


def compute_u_tau(tau_w: pt.Tensor, rho: float) -> pt.Tensor:
    return (tau_w.pow(2).sum(-1).sqrt() / rho).sqrt()

def compute_z_plus(u_tau: pt.Tensor, nu: float, nz: int = 65, z_max: float = 0.25) -> pt.Tensor:
    dz = abs(z_max / nz)
    return u_tau * dz / nu

def compute_x_plus(u_tau: pt.Tensor, nu: pt.Tensor, dx: pt.Tensor) -> pt.Tensor:
    return u_tau * dx / nu

def load_fields(load_path: str, field_names: list = None) -> dict:
    fields = {}
    field_names = field_names if field_names is not None else ["rho", "wallShearStress"]

    for f in field_names:
        loader = CSVDataloader.from_foam_surface(join(load_path, "postProcessing", "surface"), f"{f}_airfoil.raw")

        # load the field
        _tmp = loader.load_snapshot(loader.field_names[loader.write_times[0]], loader.write_times[-1])

        if len(_tmp) == 1:
            _tmp = _tmp[0]
        else:
            # if we have multiple dimensions we cat the list of tensors
            _tmp = pt.cat([t.unsqueeze(-1) for t in _tmp], dim=-1)

        # split into SS and PS
        x_camber_temp, camber_line = compute_camber_line(loader.vertices[:, 0], c=1, xf_max=0.5, f_max=0.05)
        # get all coordinates for suction and pressure side
        is_suction = loader.vertices[:, 2] > camber_line
        is_pressure = ~is_suction

        # load the grid and times once, since the same for all cases
        if "vertices_ss" not in fields.keys():
            # we don't care about the direction of the span, so take abs()
            fields["vertices_ss"] = loader.vertices[is_suction, :].abs()
            fields["vertices_ps"] = loader.vertices[is_pressure, :].abs()
            fields["write_times"] = loader.write_times[-1]

            # TODO: weights not present -> get weights
            """area_normal = -pt.vstack(loader.load_snapshot(["area_x", "area_y", "area_z"], loader.write_times[-1])).T
            area = area_normal.norm(dim=1)
            fields["area_ss"] = area[is_suction]
            fields["area_ps"] = area[is_pressure]"""

        fields[f"{f}_ss"] = _tmp[is_suction]
        fields[f"{f}_ps"] = _tmp[is_pressure]
    return fields

def plot_data(vertices_ss: pt.Tensor, vertices_ps: pt.Tensor, data_ss: pt.Tensor, data_ps: pt.Tensor, save_path, save_name, label):
    vmin = min(data_ss.min(), data_ss.min())
    vmax = max(data_ps.max(), data_ps.max())

    fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex="col")
    cf1 = ax[0].tricontourf(vertices_ss[:, 0], vertices_ss[:, 1], data_ss, vmin=vmin, vmax=vmax)
    ax[1].tricontourf(vertices_ps[:, 0], vertices_ps[:, 1], data_ps, vmin=vmin, vmax=vmax)

    ax[0].set_title(r"$\mathrm{SS}$")
    ax[1].set_title(r"$\mathrm{PS}$")
    ax[1].set_xlim(0, 1)

    ax[1].set_xlabel("$x~/~c$")
    for i in range(2):
        ax[i].set_ylabel("$y~/~c$")

    cbar = fig.colorbar(cf1, ax=ax, orientation="horizontal", shrink=0.6)
    cbar.set_label(label)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35)
    plt.savefig(join(save_path, f"{save_name}.png"))
    plt.close(fig)

if __name__ == "__main__":
    # load and save paths
    load_dir = join("/media", "janis", "Elements", "Janis", "2D_buffet_simulation", "DDES_3D_Ma0.73_Re3e6")
    save_dir = join("..", "run", "plots", "DDES_validation")
    case = r"DDES_SA_Re3e6_Ma0.73_alpha3.5deg_y65_ymax0.25"

    # flow properties
    mu = 7.7319e-05
    u_inf = 242.16629
    rho_inf = 0.957837

    # simulation setup
    chord = 1
    ny = 65
    y_max = 0.25

    # load the surface data
    data = load_fields(join(load_dir, case))

    # use latex fonts
    plt.rcParams.update({"text.usetex": True, "figure.dpi": 360})

    # create plot directory
    if not exists(save_dir):
        makedirs(save_dir)

    # compute and plot u_tau
    u_tau_ss = compute_u_tau(data["wallShearStress_ss"], rho_inf)
    u_tau_ps = compute_u_tau(data["wallShearStress_ps"], rho_inf)
    plot_data(data["vertices_ss"], data["vertices_ps"], u_tau_ss, u_tau_ps, save_dir, "u_tau",
              r"$u_\tau ~[m s^{-1}]$")

    # compute and plot delta z+
    dz_plus_ss = compute_z_plus(u_tau_ss, mu / rho_inf, nz=ny, z_max=y_max)
    dz_plus_ps = compute_z_plus(u_tau_ps, mu / rho_inf, nz=ny, z_max=y_max)
    plot_data(data["vertices_ss"], data["vertices_ps"], dz_plus_ss, dz_plus_ps, save_dir, "delta_z_plus",
              r"$\Delta z^+ ~[-]$")

    # compute and plot delta x+
    dz = y_max / ny
    # approx cell area from paraview TODO: get correct cell area
    estimator_ss = CellWeightEstimator(data["vertices_ss"], normalize=False)
    estimator_ps = CellWeightEstimator(data["vertices_ps"], normalize=False)
    data["area_ss"] = estimator_ss.weights
    data["area_ps"] = estimator_ps.weights

    u_tau_ss = compute_u_tau(data["wallShearStress_ss"], data["rho_ss"])
    u_tau_ps = compute_u_tau(data["wallShearStress_ps"], data["rho_ps"])

    dx_plus_ss = compute_x_plus(u_tau_ss, mu / data["rho_ss"], data["area_ss"] / dz)
    dx_plus_ps = compute_x_plus(u_tau_ps, mu / data["rho_ps"], data["area_ps"] / dz)
    plot_data(data["vertices_ss"], data["vertices_ps"], dx_plus_ss, dx_plus_ps, save_dir, "delta_x_plus",
              r"$\Delta x^+ ~[-]$")