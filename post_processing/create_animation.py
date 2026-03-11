"""
create an animation of the flow field along with the course of cl
"""
import torch as pt
import matplotlib.pyplot as plt

from os import makedirs
from pandas import read_csv
from os.path import join, exists
from matplotlib.patches import Polygon
from flowtorch.data import FOAMDataloader, mask_box
from matplotlib.animation import FuncAnimation, FFMpegWriter

from utils import load_force_coeffs

def prepare_data(load_path: str, bounds : list, save_path, field_name: str = "Ma") -> None:
    # prepare the forces
    forces = load_force_coeffs(load_path)
    forces = forces[["t", "cy"]]
    pt.save(forces, join(save_dir, "forces.pt"))
    del forces

    # load the snapshots, we only have 2D data (URANS)
    loader = FOAMDataloader(load_path)
    xz = loader.vertices[:, [0, 2]]
    write_times = loader.write_times[1:]
    mask = mask_box(xz, lower=bounds[0], upper=bounds[1])

    # mask the vertices
    xz = pt.stack([pt.masked_select(xz[:, d], mask) for d in range(2)], dim=1)
    data = pt.zeros((xz.shape[0], len(write_times)))

    # load the data
    for i, t in enumerate(write_times):
        data[:, i] = pt.masked_select(loader.load_snapshot(field_name, t), mask)

    # save everything
    pt.save({field_name: data, "write_times": write_times, "xz": xz}, join(save_path, f"{field_name}_fields.pt"))


if __name__ == '__main__':
    # load and save paths
    load_dir = join("/media", "janis", "Elements", "Janis", "2D_buffet_simulation", "URANS_2D_Ma0.73_Re3e6")
    save_dir = join("..", "run", "plots", "URANS_validation", "URANS_blockMesh", "SALSA", "animations")
    case = r"URANS_SALSA_alpha3.5deg_blockMesh_useRmod_useSmod"

    # settings
    prepare = False
    bounds = [[-0.25, -0.25], [5, 2]]

    # flow conditions
    chord = 1
    u_inf = 242.16629
    field_name = "Ma"

    # create plot directory
    if not exists(save_dir):
        makedirs(save_dir)

    # load and prepare daa once
    if prepare:
        prepare_data(join(load_dir, case), bounds, save_dir)
        exit()
    else:
        forces = pt.load(join(save_dir, "forces.pt"), weights_only=False)
        data = pt.load(join(save_dir, f"{field_name}_fields.pt"), weights_only=False)
        write_times = list(map(float, data["write_times"]))
        field = data[field_name]
        xz = data["xz"]
        del data

        # load the airfoil coordinates
        oat = read_csv(join("..", "grid_generation", "oat15.dat"), sep=r"\s+", skiprows=1, header=None, names=["x", "y"])

    # use latex fonts
    plt.rcParams.update({"text.usetex": True, "figure.dpi": 360})

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

        return cf

    ax.set_xlim(-0.2/chord, 2.5/chord)
    ax.set_ylim(-0.2/chord, 1/chord)

    ax.add_patch(Polygon(oat/chord, facecolor="white"))
    ax.set_xlabel("$x~/~c$")
    ax.set_ylabel("$z~/~c$")
    ax.set_aspect("equal")
    fig.tight_layout()

    ani = FuncAnimation(fig, animate, frames=field.shape[1], blit=False, repeat=True)
    writer = FFMpegWriter(fps=int(len(write_times) / 10))
    ani.save(join(save_dir, f"flow_field_animation_{field_name}.mp4"), writer=writer)
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
    yLim = (0.8, 1)

    ax[1].plot(forces["t"][idx_0:] * u_inf / chord, forces["cy"][idx_0:])
    ax[1].scatter(forces["t"][idx_0] * u_inf/chord, forces["cy"][idx_0], marker="o", color="red", zorder=10)
    ax[1].axvline(forces["t"][idx_0] * u_inf/chord, color="red", zorder=10, ls="--")
    ax[1].set_xlim(write_times[0] * u_inf/chord, write_times[-1] * u_inf/chord)
    ax[1].tick_params(axis="x", which="minor", bottom=False)
    ax[1].minorticks_on()
    ax[1].set_ylim(yLim)
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
        ax[1].set_ylim(yLim)
        ax[1].set_xlabel(r"$\tau$")
        ax[1].set_ylabel(r"$c_l$")
        ax[1].minorticks_on()

        return cf

    # create animation
    ani = FuncAnimation(fig, animate, frames=field.shape[1], blit=False, repeat=True)
    writer = FFMpegWriter(fps=int(len(write_times) / 10))
    ani.save(join(save_dir, f"flow_field_cl_animation_{field_name}.mp4"), writer=writer)
