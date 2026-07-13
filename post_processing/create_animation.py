"""
create an animation of the flow field along with the course of cl
"""
from typing import Union

import torch as pt
import matplotlib.pyplot as plt

from os import makedirs
from pandas import read_csv
from os.path import join, exists
from matplotlib.patches import Polygon, FancyArrowPatch
from flowtorch.data import FOAMDataloader, mask_box
from matplotlib.animation import FuncAnimation, FFMpegWriter

from utils import load_force_coeffs

def prepare_data(load_path: str, bounds : list, save_path, case: str, field_name: str = "Ma", dims = None, y_max = -0.0375,
                 n_dims : int = 2, t_start: Union[float, int] = 0.6) -> None:
    if dims is None:
        dims = [0, 2]

    # prepare the forces
    forces = load_force_coeffs(load_path)
    pt.save(forces[["t", "cy"]], join(save_dir, f"forces_{case}.pt"))
    del forces

    # load the snapshots of the volume data
    loader = FOAMDataloader(load_path)

    # mask the coordinates
    _coord = loader.vertices

    # apply the mask
    mask = mask_box(_coord, lower=bounds[0], upper=bounds[1])

    # mask the vertices
    _coord = pt.stack([pt.masked_select(_coord[:, d], mask) for d in range(3)], dim=1)

    # check if we have multiple cells in spanwise direction
    if n_dims == 2:
        _coord = loader.vertices[:, dims]

        # for 2D we don't need _idx, but use here so we can load the fields independently of n_dims
        _idx = pt.ones(_coord[mask].shape[0],).bool()
    else:
        # if so, extract a slice from the middle of the domain
        _idx = pt.isclose(_coord[:, 1], pt.tensor(y_max)/2)
        _coord = _coord[_idx, :][:, dims]

    # take all available write times except zero
    _write_times = [t for t in loader.write_times if float(t) >= t_start]

    # allocate a tensor for the flow fields, for the animation we don't care about DP
    if n_dims == 2:
        _data = pt.zeros((_coord[mask].shape[0], len(_write_times))).to(pt.float32)
    else:
        _data = pt.zeros((_coord.shape[0], len(_write_times))).to(pt.float32)

    # load the data, loop over times due to memory constraints for the DDES case
    print(f"Loading snapshots for field {field_name}. Found {len(_write_times)} snapshots.")
    for i, t in enumerate(_write_times):
        print(f"\rLoading write time {i + 1} / {len(_write_times)}.", end="", flush=True)
        _data[:, i] = pt.masked_select(loader.load_snapshot(field_name, t), mask)[_idx].to(pt.float32)

    # save everything
    print("Saving data.")
    pt.save({field_name: _data.to(pt.float32), "write_times": _write_times, "xz": _coord.to(pt.float32)},
            join(save_path, f"{field_name}_fields_{case}.pt"))


def compute_aoa(time_steps, aoa0: Union[int, float], amplitude: Union[int, float], frequency: Union[int, float],
                phase: Union[int, float] = 0) -> pt.Tensor:
    # verified with actual inflow angle from the simulations (in ParaView)
    return aoa0 + amplitude * pt.sin(2*pt.pi*frequency*time_steps - pt.deg2rad(pt.tensor(phase)))


if __name__ == '__main__':
    # load and save paths
    # """
    load_dir = join("/media", "janis", "Elements", "Janis", "2D_buffet_simulation",
                    "URANS_2D_Ma0.73_Re3e6_pitching_volume_data")
    save_dir = join("..", "run", "plots", "URANS_pitching", "URANS_blockMesh", "synchronization_analysis_3.5deg",
                    "animations")
    case = r"A1.75_f6"
    #"""

    # load_dir = join("/media", "janis", "Elements", "Janis", "2D_buffet_simulation", "DDES_3D_Ma0.73_Re3e6")
    # save_dir = join("..", "run", "plots", "DDES_validation", "animations")
    # case = r"DDES_SALSA_Re3e6_Ma0.73_alpha3.5deg_y65_ymax0.25"

    # settings
    prepare = False
    bounds = [[-0.25, -1, -0.25], [5, 1, 2]]

    # if we have pitching, add an arrow with the current AoA
    pitching = True
    aoa0 = 3.5
    frequency = 6
    amplitude = 1.75
    t_pitching_start = 0.06

    # 2D URANS
    n_dims = 2
    y_max = -0.0375
    t_start = 2

    # 3D DDES
    # y_max = -0.25
    # n_dims = 3          # it is assumed that the y-coord. is the spanwise direction
    # t_start = 0.15

    # flow conditions
    chord = 1
    u_inf = 242.16629

    # currently only scalar fields are supported
    field_name = "Ma"

    # create plot directory
    if not exists(save_dir):
        makedirs(save_dir)

    # load and prepare data once
    if prepare:
        if case is not None:
            prepare_data(join(load_dir, case), bounds, save_dir, case, y_max=y_max, n_dims=n_dims, t_start=t_start)
        else:
            prepare_data(load_dir, bounds, save_dir, case, y_max=y_max, n_dims=n_dims, t_start=t_start)
        exit()
    else:
        if case is not None:
            forces = pt.load(join(save_dir, f"forces_{case}.pt"), weights_only=False)
            data = pt.load(join(save_dir, f"{field_name}_fields_{case}.pt"), weights_only=False)
        else:
            forces = pt.load(join(save_dir, f"forces.pt"), weights_only=False)
            data = pt.load(join(save_dir, f"{field_name}_fields.pt"), weights_only=False)
        write_times = list(map(float, data["write_times"]))
        field = data[field_name]
        xz = data["xz"]
        del data

        # load the airfoil coordinates
        oat = read_csv(join("..", "grid_generation", "oat15.dat"), sep=r"\s+", skiprows=1, header=None, names=["x", "y"])

    # if we have pitching, compute the AoA.
    # note: this is the AoA @ inlet, but in video we don't show the inlet -> there is a delay
    if pitching:
        idx_start = pt.where(abs(pt.tensor(forces.t.values) - t_pitching_start) < 1e-8)[0][0].item()
        aoa = compute_aoa(pt.from_numpy(forces.t.values[idx_start:]), aoa0, amplitude, frequency)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True, "figure.dpi": 360})

    # set the fps, make sure to not set it to zero if we haven't enough snapshots
    fps = int(len(write_times) / 10) if len(write_times) >= 150 else 15

    # add the arrow
    if pitching:
        arrow_origin = pt.tensor([-0.2/chord, 0.0])
        arrow_length = 0.2

    # animate flow field only
    fig, ax = plt.subplots(figsize=(6, 3))
    cf = ax.tricontourf(xz[:, 0]/chord, xz[:, 1]/chord, field[:, 0], cmap="coolwarm", levels=500, extend="both",
                        vmin=0, vmax=1.4)

    # colorbar settings
    cbar = fig.colorbar(cf, ax=ax, shrink=0.6)
    cbar.set_ticks([0.2, 0.4, 0.6, 0.8, 1, 1.2])
    cbar.set_label("$~$" + field_name + "$~[-]$")


    # animate
    def animate(i):
        print("\r", f"Creating frame {i + 1:03d} / {len(write_times)}", end="")
        # update flow field
        ax.clear()
        cf = ax.tricontourf(xz[:, 0] / chord, xz[:, 1] / chord, field[:, i], cmap="coolwarm", levels=500,
                               extend="both", vmin=0, vmax=1.4)
        ax.add_patch(Polygon(oat / chord, facecolor="white"))
        ax.set_xlim(-0.2 / chord, 2.5 / chord)
        ax.set_ylim(-0.2 / chord, 1 / chord)
        ax.set_xlabel("$x~/~c$")
        ax.set_ylabel("$z~/~c$")
        ax.set_aspect("equal")
        ax.set_title(fr"$\tau = {write_times[i] * u_inf / chord:.2f}$")

        # add arrow for AoA
        if pitching:
            idx = pt.where(abs(pt.tensor(forces.t.values) - write_times[i]) < 1e-6)[0][0].item()
            dx = arrow_length * pt.cos(pt.deg2rad(aoa[idx - idx_start]))
            dz = arrow_length * pt.sin(pt.deg2rad(aoa[idx - idx_start]))

            arrow = FancyArrowPatch(posA=(arrow_origin[0], arrow_origin[1]), posB=(arrow_origin[0] + dx, arrow_origin[1] + dz),
                                    color="black", arrowstyle="->", mutation_scale=12, linewidth=2)
            ax.add_patch(arrow)

        return cf

    ax.set_xlim(-0.2/chord, 2.5/chord)
    ax.set_ylim(-0.2/chord, 1/chord)

    ax.add_patch(Polygon(oat/chord, facecolor="white"))
    ax.set_xlabel("$x~/~c$")
    ax.set_ylabel("$z~/~c$")
    ax.set_aspect("equal")
    fig.tight_layout()

    ani = FuncAnimation(fig, animate, frames=field.shape[1], blit=False, repeat=True)
    writer = FFMpegWriter(fps=fps)
    if pitching:
        ani.save(join(save_dir, f"flow_field_animation_{field_name}_{case}_with_arrow.mp4"), writer=writer)
    else:
        ani.save(join(save_dir, f"flow_field_animation_{field_name}_{case}.mp4"), writer=writer)
    plt.close(fig)

    #  ----------------------------- animate flow field with cl together -------------------------------------
    fig = plt.figure(figsize=(6, 4))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[4.82, 1])
    ax = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0])]

    # flow field plot
    cf = ax[0].tricontourf(xz[:, 0]/chord, xz[:, 1]/chord, field[:, 0], cmap="coolwarm", levels=500, extend="both",
                        vmin=0, vmax=1.4)

    # colorbar settings
    cbar = fig.colorbar(cf, ax=ax[0], shrink=0.6)
    cbar.set_ticks([0.2, 0.4, 0.6, 0.8, 1, 1.2])
    cbar.set_label("$~$" + field_name + "$~[-]$")

    ax[0].set_xlim(-0.2/chord, 2.5/chord)
    ax[0].set_ylim(-0.2/chord, 1/chord)
    ax[0].set_title(fr"$\tau = {'{:.2f}'.format(write_times[0] * u_inf / chord)}$")

    ax[0].add_patch(Polygon(oat/chord, facecolor="white"))
    ax[0].set_xlabel("$x~/~c$")
    ax[0].set_ylabel("$z~/~c$")
    ax[0].set_aspect("equal")

    # now plot the cl -> smaller portion than Ma field
    idx_0 = forces["t"][forces["t"] == write_times[0]].index.values[0]

    # increase the y-limits a little bit more than min./max. of cl in this time span to make it look nicer
    EPS = 0.05

    ax[1].plot(forces["t"][idx_0:] * u_inf / chord, forces["cy"][idx_0:])
    ax[1].scatter(forces["t"][idx_0] * u_inf/chord, forces["cy"][idx_0], marker="o", color="red", zorder=10)
    ax[1].axvline(forces["t"][idx_0] * u_inf/chord, color="red", zorder=10, ls="--")
    ax[1].set_xlim(write_times[0] * u_inf/chord, write_times[-1] * u_inf/chord)
    ax[1].tick_params(axis="x", which="minor", bottom=False)
    ax[1].minorticks_on()
    ax[1].set_ylim(forces["cy"][idx_0:].min() - EPS, forces["cy"][idx_0:].max() + EPS)
    ax[1].set_xlabel(r"$\tau$")
    ax[1].set_ylabel(r"$c_l$")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.25)

    # animate
    def animate(i):
        print("\r", f"Creating frame {i+1:03d} / {len(write_times)}", end="")
        # update flow field
        ax[0].clear()
        ax[1].clear()
        cf = ax[0].tricontourf(xz[:, 0] / chord, xz[:, 1] / chord, field[:, i], cmap="coolwarm", levels=500,
                               extend="both", vmin=0, vmax=1.4)
        ax[0].add_patch(Polygon(oat / chord, facecolor="white"))
        ax[0].set_xlim(-0.2 / chord, 2.5 / chord)
        ax[0].set_ylim(-0.2 / chord, 1 / chord)
        ax[0].set_xlabel("$x~/~c$")
        ax[0].set_ylabel("$z~/~c$")
        ax[0].set_aspect("equal")
        ax[0].set_title(fr"$\tau = {write_times[i] * u_inf / chord:.2f}$")

        # update c_l plot
        idx = forces["t"][forces["t"] == write_times[i]].index.values[0]
        ax[1].plot(forces["t"][idx_0:] * u_inf / chord, forces["cy"][idx_0:])
        ax[1].scatter(forces["t"][idx] * u_inf / chord, forces["cy"][idx], color="red", zorder=10)
        ax[1].axvline(forces["t"][idx] * u_inf / chord, color="red", ls="--", zorder=10)
        ax[1].set_xlim(write_times[0] * u_inf / chord, write_times[-1] * u_inf / chord)
        ax[1].set_ylim(forces["cy"][idx_0:].min() - EPS, forces["cy"][idx_0:].max() + EPS)
        ax[1].set_xlabel(r"$\tau$")
        ax[1].set_ylabel(r"$c_l$")
        ax[1].minorticks_on()

        # add arrow for AoA
        if pitching:
            idx = pt.where(abs(pt.tensor(forces.t.values) - write_times[i]) < 1e-6)[0][0].item()
            dx = arrow_length * pt.cos(pt.deg2rad(aoa[idx - idx_start]))
            dz = arrow_length * pt.sin(pt.deg2rad(aoa[idx - idx_start]))

            arrow = FancyArrowPatch(posA=(arrow_origin[0], arrow_origin[1]), posB=(arrow_origin[0] + dx, arrow_origin[1] + dz),
                                    color="black", arrowstyle="->", mutation_scale=12, linewidth=2)
            ax[0].add_patch(arrow)
        return cf

    # create animation
    ani = FuncAnimation(fig, animate, frames=field.shape[1], blit=False, repeat=True)
    writer = FFMpegWriter(fps=fps)
    if pitching:
        ani.save(join(save_dir, f"flow_field_cl_animation_{field_name}_{case}_with_arrow.mp4"), writer=writer)
    else:
        ani.save(join(save_dir, f"flow_field_cl_animation_{field_name}_{case}.mp4"), writer=writer)
