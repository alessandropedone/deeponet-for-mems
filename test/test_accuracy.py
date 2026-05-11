from pathlib import Path
import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

from utils import cantilever_shape_np, compute_displacement_from_history

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-simulation", action="store_true")
    ap.add_argument("--big-deformation", action="store_true")

    workdir_ref = Path(f"temp/run_nmodes_4") if not ap.parse_args().big_deformation else Path(f"temp/run_nmodes_4_big_deformation")

    cmd = [
        "python",
        "-m",
        "src.multi_physics.solver",
        "--template-geo",
        "geometries/cantilever1.geo" if not ap.parse_args().big_deformation else "geometries/cantilever2.geo",
        "--workdir",
        str(workdir_ref),
        "--nmodes",
        "4",
        "--dt",
        "1e-5",
        "--nsteps",
        "40",
        "--Vdc",
        "0",
        "--Vac",
        "5" if not ap.parse_args().big_deformation else "230",
        "--freq",
        "2.5e3",
        "--Vupper",
        "0",
        "--Vouter",
        "0",
        "--omega",
        "6.3e5",
        "3.9e6",
        "1.1e7",
        "2.1e7",
        "--mass",
        "1e-12",
        "1e-12",
        "1e-12",
        "1e-12",
        "--zeta",
        "0.01",
        "0.01",
        "0.01",
        "0.01",
        "--print-every",
        "1",
        "--fail-fast",
    ]

    if not ap.parse_args().no_simulation:
        print(f"Running classical ROM simulation with 4 modes...")
        subprocess.run(cmd, check=True)

    workdir = Path(f"temp/run_nmodes_4_nn") if not ap.parse_args().big_deformation else Path(f"temp/run_nmodes_4_nn_big_deformation")

    cmd = [
        "python",
        "-m",
        "src.multi_physics.solver",
        "--template-geo",
        "geometries/cantilever1.geo" if not ap.parse_args().big_deformation else "geometries/cantilever2.geo",
        "--workdir",
        str(workdir),
        "--nmodes",
        "4",
        "--dt",
        "1e-5",
        "--nsteps",
        "40",
        "--Vdc",
        "0",
        "--Vac",
        "5" if not ap.parse_args().big_deformation else "230",
        "--freq",
        "2.5e3",
        "--Vupper",
        "0",
        "--Vouter",
        "0",
        "--omega",
        "6.3e5",
        "3.9e6",
        "1.1e7",
        "2.1e7",
        "--mass",
        "1e-12",
        "1e-12",
        "1e-12",
        "1e-12",
        "--zeta",
        "0.01",
        "0.01",
        "0.01",
        "0.01",
        "--print-every",
        "1",
        "--fail-fast",
        "--derivative-nn-path",
        "models/derivative1.keras" if not ap.parse_args().big_deformation else "models/derivative2.keras",
    ]

    if not ap.parse_args().no_simulation:
        print(f"Running DL-ROM simulation with 4 modes...")
        subprocess.run(cmd, check=True)

    # Compare the L2 norms of the displacement fields
    L_m = 1e-4

    x = np.linspace(0, L_m, 100)
    u = compute_displacement_from_history(4, x, L_m, workdir)
    u_ref = compute_displacement_from_history(4, x, L_m, workdir_ref)
    diff = u - u_ref

    # Integrate in L2 the difference over the length of the beam (row-wise)
    # and then integrate it over time (column-wise)
    l2_norm = np.sqrt(np.trapezoid(diff**2, x))
    # Do the same for the reference solution
    l2_norm_ref = np.sqrt(np.trapezoid(u_ref**2, x))
    print("")
    print("DL-ROM vs classical ROM")
    print("="*70)
    print("")
    print("L2 norm in space of the displacement fields")
    print(f"{'-'*70}")
    print(f"{'L2 norm range (Classical ROM)':45s} ({l2_norm_ref.min():.3e}, {l2_norm_ref.max():.3e})")
    print(f"{'L2 norm of difference range':45s} ({l2_norm.min():.3e}, {l2_norm.max():.3e})")
    print(f"{'-'*70}")
    print("")
    print("Linf norm in space of the displacement fields")
    print(f"{'-'*70}")
    print(f"{'Linf norm range (Classical ROM)':45s} ({np.max(abs(u_ref), axis=1).min():.3e}, {np.max(abs(u_ref), axis=1).max():.3e})")
    print(f"{'Linf norm of difference range':45s} ({np.max(abs(diff), axis=1).min():.3e}, {np.max(abs(diff), axis=1).max():.3e})")
    print(f"{'-'*70}")
    print("")
    print("Capacitance")
    print(f"{'-'*70}")
    csv_path = workdir / "modal_history.csv"
    csv_path_ref = workdir_ref / "modal_history.csv"
    data = pd.read_csv(csv_path)
    data_ref = pd.read_csv(csv_path_ref)
    capacity = data["cap_like_F"].values
    capacity_ref = data_ref["cap_like_F"].values
    capacity = np.nan_to_num(capacity, nan=1e-30)  # Avoid division by zero
    capacity_ref = np.nan_to_num(capacity_ref, nan=1e-30)  # Avoid division by zero
    capacity_diff = abs(capacity - capacity_ref)
    print(f"{'Capacitance range (Classical ROM)':45s}  ({capacity_ref[1:].min():.3e}, {capacity_ref[1:].max():.3e})")
    print(f"{'Capacitance difference range':45s}  ({capacity_diff[1:].min():.3e}, {capacity_diff[1:].max():.3e})")
    print("")

    # Plot all capacities over time
    time = data["t_s"].values
    plt.figure(figsize=(10, 6))
    plt.plot(time[1:], capacity[1:], label="DL-ROM", linestyle="-", linewidth=2)
    plt.plot(time[1:], capacity_ref[1:], label="Classical ROM", linestyle="--", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Capacitance (F)")
    plt.title("Capacitance over time for different numbers of modes")
    plt.legend()
    plt.grid()

    # Global y-limits so the plot doesn't jump around
    ymin = min(u.min(), u_ref.min())
    ymax = max(u.max(), u_ref.max())

    fig, ax = plt.subplots(figsize=(10, 6))

    (line1,) = ax.plot([], [], label="DL-ROM", linestyle="-", linewidth=2)
    (line2,) = ax.plot([], [], label="Classical ROM", linestyle="--", linewidth=2)
    (line3,) = ax.plot([], [], label="Difference", linestyle=":", linewidth=2)

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Space (m)")
    ax.set_ylabel("Displacement (m)")
    title = ax.set_title("Beam displacement at t = 0 μs")
    ax.grid()
    ax.legend(loc="lower left")

    def update(frame):
        line1.set_data(x, u[frame, :])
        line2.set_data(x, u_ref[frame, :])
        line3.set_data(x, diff[frame, :])

        title.set_text(f"Beam displacement at t = {time[frame]*1e6:.0f} μs")

        return line1, line2, line3, title

    anim = FuncAnimation(
        fig,
        update,
        frames=len(time),
        interval=200,  # ms between frames
        blit=False,
    )
    
    plt.show()


if __name__ == "__main__":
    main()
