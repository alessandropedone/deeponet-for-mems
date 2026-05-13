from pathlib import Path
import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import re

from utils import compute_displacement_from_history


def main():

    # ----------------------------
    # Parse command line arguments
    # ----------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-simulation", action="store_true")
    ap.add_argument("--big-deformation", action="store_true")
    ap.add_argument("--accurate-capacity", action="store_true")

    # -----------------------------------------
    # Run the reference simulation with 4 modes
    # -----------------------------------------
    workdir_ref = (
        Path(f"temp/run_nmodes_4")
        if not ap.parse_args().big_deformation
        else Path(f"temp/run_nmodes_4_big_deformation")
    )
    cmd = [
        "python",
        "-m",
        "src.multi_physics.solver",
        "--template-geo",
        (
            "geometries/cantilever1.geo"
            if not ap.parse_args().big_deformation
            else "geometries/cantilever2.geo"
        ),
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

    # ---------------------------------------
    # Run the DL-ROM simulation with 4 modes
    # ---------------------------------------
    workdir = (
        Path(f"temp/run_nmodes_4_nn")
        if not ap.parse_args().big_deformation
        else Path(f"temp/run_nmodes_4_nn_big_deformation")
    )
    cmd = [
        "python",
        "-m",
        "src.multi_physics.solver",
        "--template-geo",
        (
            "geometries/cantilever1.geo"
            if not ap.parse_args().big_deformation
            else "geometries/cantilever2.geo"
        ),
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
        (
            "models/derivative1.keras"
            if not ap.parse_args().big_deformation
            else "models/derivative2.keras"
        ),
    ]
    if not ap.parse_args().accurate_capacity:
        cmd.append("--no-postprocessing")
    else:
        cmd.append("--postprocessing-step")
        cmd.append("5")
        
    if not ap.parse_args().no_simulation:
        print(f"Running DL-ROM simulation with 4 modes...")
        subprocess.run(cmd, check=True)

    # ---------------------------------------------------------
    # Compare the displacement fields and capacitance over time
    # ---------------------------------------------------------
    L_m = 1e-4
    x = np.linspace(0, L_m, 100)
    u = compute_displacement_from_history(4, x, L_m, workdir)
    u_ref = compute_displacement_from_history(4, x, L_m, workdir_ref)
    diff = u - u_ref

    # --- L2 and Linf norms ---
    # Integrate in L2 the difference over the length of the beam (row-wise)
    l2_norm = np.sqrt(np.trapezoid(diff**2, x))
    # Do the same for the reference solution
    l2_norm_ref = np.sqrt(np.trapezoid(u_ref**2, x))
    print("")
    print("DL-ROM vs classical ROM")
    print("=" * 75)
    print("")
    print("L2 norm in space of the displacement fields")
    print(f"{'-'*75}")
    print(
        f"{'L2 norm range (Classical ROM)':50s} ({l2_norm_ref.min():.3e}, {l2_norm_ref.max():.3e})"
    )
    print(
        f"{'L2 norm of difference range':50s} ({l2_norm.min():.3e}, {l2_norm.max():.3e})"
    )

    # --- Linf norm ---
    print("")
    print("Linf norm in space of the displacement fields")
    print(f"{'-'*75}")
    print(
        f"{'Linf norm range (Classical ROM)':50s} ({np.max(abs(u_ref), axis=1).min():.3e}, {np.max(abs(u_ref), axis=1).max():.3e})"
    )
    print(
        f"{'Linf norm of difference range':50s} ({np.max(abs(diff), axis=1).min():.3e}, {np.max(abs(diff), axis=1).max():.3e})"
    )
    print("")

    # --- Capacitance ---
    print("Capacitance")
    print(f"{'-'*75}")
    csv_path = workdir / "modal_history.csv"
    csv_path_ref = workdir_ref / "modal_history.csv"
    data = pd.read_csv(csv_path)
    data_ref = pd.read_csv(csv_path_ref)
    thickness_m = 10e-6
    capacity = thickness_m * data["cap_like_F_approx"].values
    capacity_ref = thickness_m * data_ref["cap_like_F"].values
    capacity = np.nan_to_num(capacity, nan=1e-30)  # Avoid division by zero
    capacity_ref = np.nan_to_num(capacity_ref, nan=1e-30)  # Avoid division by zero
    capacity_diff = abs(capacity - capacity_ref)
    print(
        f"{'Cap. range (Classical ROM)':50s}  ({capacity_ref[1:].min():.3e}, {capacity_ref[1:].max():.3e})"
    )
    print(
        f"{'Cap. difference range':50s}  ({capacity_diff[1:].min():.3e}, {capacity_diff[1:].max():.3e})"
    )
    print("")

    # -----------------------
    # Analyze execution times
    # -----------------------
    print("Total execution time and speedup")
    print(f"{'-'*75}")
    execution_time_path = workdir / "execution_time.csv"
    execution_time_path_ref = workdir_ref / "execution_time.csv"
    exec_time_data = pd.read_csv(execution_time_path)
    exec_time_data_ref = pd.read_csv(execution_time_path_ref)
    total_time = exec_time_data["total_s"].values[0]
    total_time_ref = exec_time_data_ref["total_s"].values[0]
    print(f"{'Total runtime (DL-ROM)':50s} {total_time:.2f} seconds")
    print(f"{'Total runtime (Classical ROM)':50s} {total_time_ref:.2f} seconds")
    print(f"{'Speedup':50s} {total_time_ref/total_time:.2f}x")
    print("")
    print("Postprocessing/meshing time")
    print(f"{'-'*75}")
    print(f"{'Postprocessing/meshing time (DL-ROM)':50s} {exec_time_data['postprocessing_and_meshing_s'].values[0]:.2f} seconds")
    print(f"{'Postprocessing/meshing time (Classical ROM)':50s} {exec_time_data_ref['postprocessing_and_meshing_s'].values[0]:.2f} seconds")
    print("")
    print("Solution time")
    print(f"{'-'*75}")
    print(f"{'Solution time (DL-ROM)':50s} {exec_time_data['solution_s'].values[0]:.2f} seconds")
    print(f"{'Solution time (Classical ROM)':50s} {exec_time_data_ref['solution_s'].values[0]:.2f} seconds")
    print("")

    # ------------------------------
    # Plot the capacitance over time
    # ------------------------------
    # Time vector
    time = data["t_s"].values
    plt.figure(figsize=(10, 6))
    plt.plot(time[1:], capacity[1:], label="DL-ROM", linestyle="-", linewidth=2)
    plt.plot(
        time[1:], capacity_ref[1:], label="Classical ROM", linestyle="--", linewidth=2
    )
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
