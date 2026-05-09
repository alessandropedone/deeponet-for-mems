from pathlib import Path
import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse


def cantilever_shape_np(xi, beta, L) -> np.ndarray:
    """Compute the shape of the cantilever beam for a given beta and length."""
    C = (np.cosh(beta * L) + np.cos(beta * L)) / (np.sinh(beta * L) + np.sin(beta * L))
    return (
        np.cosh(beta * xi)
        - np.cos(beta * xi)
        - C * (np.sinh(beta * xi) - np.sin(beta * xi))
    )


def compute_displacement_field(nmodes, x, L_m, path) -> np.ndarray:
    """Compute the displacement field from the modal history CSV file."""
    roots = np.array(
        [1.875104068711961, 4.694091132974174, 7.854757438237612, 10.995540734875466],
        dtype=float,
    )
    betas = roots / L_m
    modes = np.array([cantilever_shape_np(x, betas[i], L_m) for i in range(nmodes)])
    workdir = Path(f"{path}")
    csv_path = workdir / "modal_history.csv"
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    q_values = data[:, 3 : 3 + nmodes]
    u = q_values @ modes
    return u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-simulation", action="store_true")

    workdir = Path(f"temp/run_nmodes_4")

    cmd = [
        "python",
        "-m",
        "src.multi_physics.solver",
        "--template-geo",
        "geometries/cantilever1.geo",
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
        "5",
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
        print(f"Running with 4 modes...")
        subprocess.run(cmd, check=True)

    workdir_ref = Path(f"temp/run_nmodes_4_nn")

    cmd = [
        "python",
        "-m",
        "src.multi_physics.solver",
        "--template-geo",
        "geometries/cantilever1.geo",
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
        "5",
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
        "--nn-path",
        "models/derivative1.keras",
    ]

    if not ap.parse_args().no_simulation:
        print(f"Running with 4 modes...")
        subprocess.run(cmd, check=True)

    # Compare the L2 norms of the displacement fields
    L_m = 1e-4

    x = np.linspace(0, L_m, 100)
    u = compute_displacement_field(4, x, L_m, "temp/run_nmodes_4_nn")
    u_ref = compute_displacement_field(4, x, L_m, "temp/run_nmodes_4")
    diff = u - u_ref

    # Integrate in L2 the difference over the length of the beam (row-wise)
    # and then integrate it over time (column-wise)
    l2_norm = np.sqrt(np.trapezoid(np.trapezoid(diff**2, x), dx=1e-5))
    # Do the same for the reference solution
    l2_norm_ref = np.sqrt(np.trapezoid(np.trapezoid(u_ref**2, x), dx=1e-5))
    print("")
    print("DL-ROM vs classical ROM")
    print("")
    print("L2 norm of the displacement fields:")
    print(f"{'L2 norm of difference':35s} {l2_norm:16.6e}")
    print(f"{'L2 norm of reference':35s} {l2_norm_ref:16.6e}")
    print(f"{'Relative L2 error':35s} {l2_norm / l2_norm_ref:16.6e}")
    print("")
    print("Capacitance difference at final time step:")
    csv_path = workdir / "modal_history.csv"
    csv_path_ref = workdir_ref / "modal_history.csv"
    data = pd.read_csv(csv_path)
    data_ref = pd.read_csv(csv_path_ref)
    capacity = data["cap_like_F"].values
    capacity_ref = data_ref["cap_like_F"].values
    capacity = np.nan_to_num(capacity, nan=1e-30)  # Avoid division by zero
    capacity_ref = np.nan_to_num(capacity_ref, nan=1e-30)  # Avoid division by zero
    capacity_diff = abs(capacity - capacity_ref)
    print(f"{'Max capacitance (nmodes)':35s}  {capacity[1:].max():15.6e}")
    print(f"{'Max capacitance (reference)':35s}  {capacity_ref[1:].max():15.6e}")
    print(f"{'Max difference in capacitance':35s}  {capacity_diff[1:].max():15.6e}")
    print(
        f"{'Relative difference in capacitance':35s}  {(capacity_diff[1:]/capacity_ref[1:]).max():15.6e}"
    )
    print(f"{'-'*60}")
    print("")

    # Plot all capacities over time
    time = data["t_s"].values
    plt.figure(figsize=(10, 6))
    plt.plot(time[1:], capacity[1:], label="1 mode", linestyle="-", linewidth=2)
    plt.plot(time[1:], capacity_ref[1:], label="2 modes", linestyle="--", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Capacitance (F)")
    plt.title("Capacitance over time for different numbers of modes")
    plt.legend()
    plt.grid()

    # Global y-limits so the plot doesn't jump around
    ymin = min(u.min(), u_ref.min())
    ymax = max(u.max(), u_ref.max())

    fig, ax = plt.subplots(figsize=(10, 6))

    (line1,) = ax.plot([], [], label="Classical ROM", linestyle="-", linewidth=2)
    (line2,) = ax.plot([], [], label="DL-ROM", linestyle="--", linewidth=2)

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

        title.set_text(f"Beam displacement at t = {time[frame]*1e6:.0f} μs")

        return line1, line2, title

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
