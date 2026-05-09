"""Test the effect of the number of modes on the solution."""

import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils import cantilever_shape_np, compute_displacement_from_history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-simulation", action="store_true")

    modes_to_test = [1, 2, 3, 4]

    if not ap.parse_args().no_simulation:
        for nmodes in modes_to_test:

            workdir = Path(f"temp/run_nmodes_{nmodes}")

            cmd = [
                "python",
                "-m",
                "src.multi_physics.solver",
                "--template-geo",
                "geometries/cantilever1.geo",
                "--workdir",
                str(workdir),
                "--nmodes",
                str(nmodes),
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

            print(f"Running with {nmodes} modes...")
            subprocess.run(cmd, check=True)

    # Compare the L2 norms of the displacement fields
    L_m = 1e-4

    # read modal_history.csv and compare the L2 norms of the displacement fields
    print("")
    for nmodes in [2, 3, 4]:
        for ref in range(1, nmodes):
            x = np.linspace(0, L_m, 100)
            workdir = Path(f"temp/run_nmodes_{nmodes}")
            workdir_ref = Path(f"temp/run_nmodes_{ref}")
            u = compute_displacement_from_history(nmodes, x, L_m, workdir)
            u_ref = compute_displacement_from_history(ref, x, L_m, workdir_ref)  # Use ref modes as reference
            diff = u - u_ref
            # Integrate in L2 the difference over the length of the beam (row-wise)
            # and then integrate it over time (column-wise)
            l2_norm = np.sqrt(np.trapezoid(np.trapezoid(diff**2, x), dx=1e-5))
            # Do the same for the reference solution
            l2_norm_ref = np.sqrt(np.trapezoid(np.trapezoid(u_ref**2, x), dx=1e-5))
            print(f"Comparing {nmodes} modes to {ref} (reference) modes:")
            print("")
            print("L2 norm of the displacement fields:")
            print(f"{'L2 norm of difference':35s} {l2_norm:16.6e}")
            print(f"{'L2 norm of reference':35s} {l2_norm_ref:16.6e}")
            print(f"{'Relative L2 error':35s} {l2_norm / l2_norm_ref:16.6e}")
            print("")
            print("Capacitance difference at final time step:")
            # Read the capacitance at the final time step for both cases
            workdir_nmodes = Path(f"temp/run_nmodes_{nmodes}")
            workdir_ref = Path(f"temp/run_nmodes_{ref}")
            csv_path_nmodes = workdir_nmodes / "modal_history.csv"
            csv_path_ref = workdir_ref / "modal_history.csv"
            data_nmodes = pd.read_csv(csv_path_nmodes)
            data_ref = pd.read_csv(csv_path_ref)
            capacity_nmodes = data_nmodes["cap_like_F"].values
            capacity_ref = data_ref["cap_like_F"].values
            capacity_nmodes = np.nan_to_num(capacity_nmodes, nan=1e-30)  # Avoid division by zero
            capacity_ref = np.nan_to_num(capacity_ref, nan=1e-30)  # Avoid division by zero
            capacity_diff = abs(capacity_nmodes - capacity_ref)
            print(f"{'Max capacitance (nmodes)':35s}  {capacity_nmodes.max():15.6e}")
            print(f"{'Max capacitance (reference)':35s}  {capacity_ref.max():15.6e}")
            print(f"{'Max difference in capacitance':35s}  {capacity_diff.max():15.6e}")
            print(f"{'Relative difference in capacitance':35s}  {(capacity_diff/capacity_ref).max():15.6e}")
            print(f"{'-'*60}")
            print("")

    # Read data
    workdir_1 = Path(f"temp/run_nmodes_1")
    workdir_2 = Path(f"temp/run_nmodes_2")
    workdir_3 = Path(f"temp/run_nmodes_3")
    workdir_4 = Path(f"temp/run_nmodes_4")
    csv_path_1 = workdir_1 / "modal_history.csv"
    csv_path_2 = workdir_2 / "modal_history.csv"
    csv_path_3 = workdir_3 / "modal_history.csv"
    csv_path_4 = workdir_4 / "modal_history.csv"
    data_1 = pd.read_csv(csv_path_1)
    data_2 = pd.read_csv(csv_path_2)
    data_3 = pd.read_csv(csv_path_3)
    data_4 = pd.read_csv(csv_path_4)
    time = data_1["t_s"].values
    capacity_1 = data_1["cap_like_F"].values
    capacity_2 = data_2["cap_like_F"].values
    capacity_3 = data_3["cap_like_F"].values
    capacity_4 = data_4["cap_like_F"].values

    # Plot all capacities over time
    plt.figure(figsize=(10, 6))
    plt.plot(time, capacity_1, label="1 mode", linestyle="-",  linewidth=2)
    plt.plot(time, capacity_2, label="2 modes", linestyle="--", linewidth=2)
    plt.plot(time, capacity_3, label="3 modes", linestyle="-.", linewidth=2)
    plt.plot(time, capacity_4, label="4 modes", linestyle=":",  linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Capacitance (F)")
    plt.title("Capacitance over time for different numbers of modes")
    plt.legend()
    plt.grid()

    # Displacement fields: shape = (ntime, nx)
    u_1 = compute_displacement_from_history(1, x, L_m, workdir_1)
    u_2 = compute_displacement_from_history(2, x, L_m, workdir_2)
    u_3 = compute_displacement_from_history(3, x, L_m, workdir_3)
    u_4 = compute_displacement_from_history(4, x, L_m, workdir_4)

    # Global y-limits so the plot doesn't jump around
    ymin = min(u_1.min(), u_2.min(), u_3.min(), u_4.min())
    ymax = max(u_1.max(), u_2.max(), u_3.max(), u_4.max())

    fig, ax = plt.subplots(figsize=(10, 6))

    line1, = ax.plot([], [], label="1 mode", linestyle="-",  linewidth=2)
    line2, = ax.plot([], [], label="2 modes", linestyle="--", linewidth=2)
    line3, = ax.plot([], [], label="3 modes", linestyle="-.", linewidth=2)
    line4, = ax.plot([], [], label="4 modes", linestyle=":",  linewidth=2)
    line5, = ax.plot([], [], label="2 modes - 1 mode", linestyle="-",  linewidth=2)

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Space (m)")
    ax.set_ylabel("Displacement (m)")
    title = ax.set_title("Beam displacement at t = 0 μs")
    ax.grid()
    ax.legend(loc="lower left")


    def update(frame):
        line1.set_data(x, u_1[frame, :])
        line2.set_data(x, u_2[frame, :])
        line3.set_data(x, u_3[frame, :])
        line4.set_data(x, u_4[frame, :])
        line5.set_data(x, u_2[frame, :] - u_1[frame, :])

        title.set_text(f"Beam displacement at t = {time[frame]*1e6:.0f} μs")

        return line1, line2, line3, line4, line5, title


    anim = FuncAnimation(
        fig,
        update,
        frames=len(time),
        interval=200,   # ms between frames
        blit=False,
    )

    # --- Error fields ---
    err_21 = u_2 - u_1
    err_32 = u_3 - u_2
    err_43 = u_4 - u_3

    # Global y-limits for stability
    ymin_e = min(err_21.min(), err_32.min(), err_43.min())
    ymax_e = max(err_21.max(), err_32.max(), err_43.max())

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    e1, = ax2.plot([], [], label="2 modes - 1 mode", linestyle="-", linewidth=2)
    e2, = ax2.plot([], [], label="3 modes - 2 modes", linestyle="--", linewidth=2)
    e3, = ax2.plot([], [], label="4 modes - 3 modes", linestyle="-.", linewidth=2)

    ax2.set_xlim(x[0], x[-1])
    ax2.set_ylim(ymin_e, ymax_e)
    ax2.set_xlabel("Space (m)")
    ax2.set_ylabel("Error (m)")
    title2 = ax2.set_title("Modal error at t = 0 μs")
    ax2.grid()
    ax2.legend(loc="lower left")


    def update_err(frame):
        e1.set_data(x, err_21[frame, :])
        e2.set_data(x, err_32[frame, :])
        e3.set_data(x, err_43[frame, :])

        title2.set_text(f"Modal error at t = {time[frame]*1e6:.0f} μs")

        return e1, e2, e3, title2


    anim_err = FuncAnimation(
        fig2,
        update_err,
        frames=len(time),
        interval=200,
        blit=False,
    )

    plt.show()

    # Save video
    # anim.save("beam_modes.mp4", fps=20)
    plt.show()


if __name__ == "__main__":
    main()
