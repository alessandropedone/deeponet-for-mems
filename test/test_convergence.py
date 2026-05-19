from pathlib import Path
import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import argparse

from utils import compute_displacement_from_history


def main():

    # ----------------------------
    # Parse command line arguments
    # ----------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-simulation", action="store_true")
    ap.add_argument("--big-deformation", action="store_true")
    ap.add_argument("--clamped", action="store_true")
    ap.add_argument("--save-frames", action="store_true")
    if ap.parse_args().big_deformation and ap.parse_args().clamped:
        raise ValueError(
            "The --big-deformation and --clamped options cannot be used together, as the current clamped beam model is not designed for large deformations."
        )

    # ----------------------------------------------------------------
    # Run the simulation with 4 modes with dt = 1e-4, 1e-5, 5e-6, 1e-6
    # ----------------------------------------------------------------
    dt = [1e-5, 5e-6, 1e-6]
    steps = [4 * int(1e-4 / dt_i) for dt_i in dt]
    for dt_i, steps_i in zip(dt, steps):
        workdir = (
            Path(f"temp/convergence/run_nmodes_4")
            if not ap.parse_args().big_deformation
            else Path(f"temp/convergence/run_nmodes_4_big_deformation")
        )
        if ap.parse_args().clamped:
            workdir = workdir.with_name(workdir.name + "_clamped")
        workdir = workdir.with_name(workdir.name + f"_dt_{dt_i:.0e}")
        cmd = [
            "python",
            "-m",
            "src.multi_physics.solver",
            "--template-geo",
            (
                "geometries/cantilever1.geo"
                if not ap.parse_args().big_deformation and not ap.parse_args().clamped
                else (
                    "geometries/clamped.geo"
                    if ap.parse_args().clamped
                    else "geometries/cantilever2.geo"
                )
            ),
            "--workdir",
            str(workdir),
            "--nmodes",
            "4",
            "--dt",
            str(dt_i),
            "--nsteps",
            str(steps_i),
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
        if ap.parse_args().clamped:
            cmd.append("--clamped")
        if not ap.parse_args().no_simulation:
            print(f"Running simulation with dt = {dt_i:.0e}...")
            subprocess.run(cmd, check=True)

    workdir = (
        Path(f"temp/convergence/run_nmodes_4")
        if not ap.parse_args().big_deformation
        else Path(f"temp/convergence/run_nmodes_4_big_deformation")
    )
    workdir = workdir.with_name(workdir.name + f"_dt_" + str(1e-7))
    if ap.parse_args().clamped:
        workdir = workdir.with_name(workdir.name + "_clamped")
    cmd = [
        "python",
        "-m",
        "src.multi_physics.solver",
        "--template-geo",
        (
            "geometries/cantilever1.geo"
            if not ap.parse_args().big_deformation and not ap.parse_args().clamped
            else (
                "geometries/clamped.geo"
                if ap.parse_args().clamped
                else "geometries/cantilever2.geo"
            )
        ),
        "--workdir",
        str(workdir),
        "--nmodes",
        "4",
        "--dt",
        "1e-7",
        "--nsteps",
        str(4 * int(1e-4 / 1e-7)),
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
            if not ap.parse_args().big_deformation and not ap.parse_args().clamped
            else (
                "models/derivative3.keras"
                if ap.parse_args().clamped
                else "models/derivative2.keras"
            )
        ),
        "--no-postprocessing",
    ]
    if ap.parse_args().clamped:
        cmd.append("--clamped")

    if not ap.parse_args().no_simulation:
        print(f"Running DL-ROM simulation with 4 modes...")
        subprocess.run(cmd, check=True)

    # ---------------------------------------------------------
    # Compare the displacement fields and capacitance over time
    # ---------------------------------------------------------

    workdir = (
        Path(f"temp/convergence/run_nmodes_4")
        if not ap.parse_args().big_deformation
        else Path(f"temp/convergence/run_nmodes_4_big_deformation")
    )
    L_m = 1e-4
    x = np.linspace(0, L_m, 100)
    u1 = compute_displacement_from_history(
        4,
        x,
        L_m,
        workdir.with_name(workdir.name + f"_dt_" + str(dt[0])),
        clamped=ap.parse_args().clamped,
    )
    u2 = compute_displacement_from_history(
        4,
        x,
        L_m,
        workdir.with_name(workdir.name + f"_dt_" + str(dt[1])),
        clamped=ap.parse_args().clamped,
    )
    u3 = compute_displacement_from_history(
        4,
        x,
        L_m,
        workdir.with_name(workdir.name + f"_dt_" + str(dt[2])),
        clamped=ap.parse_args().clamped,
    )
    u4 = compute_displacement_from_history(
        4,
        x,
        L_m,
        workdir.with_name(workdir.name + f"_dt_" + str(1e-7)),
        clamped=ap.parse_args().clamped,
    )

    thickness_m = 10e-6

    csv_path = (
        workdir.with_name(workdir.name + f"_dt_" + str(dt[0])) / "modal_history.csv"
    )
    data = pd.read_csv(csv_path)
    capacity1 = thickness_m * data["cap_like_F"].values
    time1 = data["t_s"].values
    csv_path = (
        workdir.with_name(workdir.name + f"_dt_" + str(dt[1])) / "modal_history.csv"
    )
    data = pd.read_csv(csv_path)
    capacity2 = thickness_m * data["cap_like_F"].values
    time2 = data["t_s"].values
    csv_path = (
        workdir.with_name(workdir.name + f"_dt_" + str(dt[2])) / "modal_history.csv"
    )
    data = pd.read_csv(csv_path)
    capacity3 = thickness_m * data["cap_like_F"].values
    time3 = data["t_s"].values
    csv_path = (
        workdir.with_name(workdir.name + f"_dt_" + str(1e-7)) / "modal_history.csv"
    )
    data = pd.read_csv(csv_path)
    capacity4 = thickness_m * data["cap_like_F_approx"].values
    time4 = data["t_s"].values

    # ------------------------------
    # Plot the capacitance over time
    # ------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(
        time1[1:], capacity1[1:], label="dt = " + str(dt[0]), linestyle="-", linewidth=2
    )
    plt.plot(
        time2[1:],
        capacity2[1:],
        label="dt = " + str(dt[1]),
        linestyle="--",
        linewidth=2,
    )
    plt.plot(
        time3[1:], capacity3[1:], label="dt = " + str(dt[2]), linestyle=":", linewidth=2
    )
    plt.plot(
        time4[1:],
        capacity4[1:],
        label="dt = " + str(1e-7),
        linestyle="-.",
        linewidth=2,
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Capacitance (F)")
    plt.title("Capacitance over time for different numbers of modes")
    plt.legend()
    plt.grid()

    # Save this plot
    if ap.parse_args().save_frames:
        plt.savefig("temp/convergence/capacitance_over_time.png", dpi=300)

    # Global y-limits so the plot doesn't jump around
    ymin = min(u1.min(), u2.min(), u3.min(), u4.min())
    ymax = max(u1.max(), u2.max(), u3.max(), u4.max())

    fig, ax = plt.subplots(figsize=(10, 6))

    (line1,) = ax.plot([], [], label="dt = " + str(dt[0]), linestyle="-", linewidth=2)
    (line2,) = ax.plot([], [], label="dt = " + str(dt[1]), linestyle="--", linewidth=2)
    (line3,) = ax.plot([], [], label="dt = " + str(dt[2]), linestyle=":", linewidth=2)
    (line4,) = ax.plot([], [], label="dt = " + str(1e-7), linestyle="-.", linewidth=2)

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Space (m)")
    ax.set_ylabel("Displacement (m)")
    title = ax.set_title("Beam displacement at t = 0 μs")
    ax.grid()
    ax.legend(loc="lower left")

    step = 4 * int(1e-4 / 1e-7)

    def update(frame):
        i1 = int(frame * steps[0] / steps[2])
        i2 = int(frame * steps[1] / steps[2])
        i4 = int(frame * step / steps[2])
        line1.set_data(x, u1[i1, :])
        line2.set_data(x, u2[i2, :])
        line3.set_data(x, u3[frame, :])
        line4.set_data(x, u4[i4, :])

        title.set_text(f"Beam displacement at t = {time3[frame]:.2e} s")

        return line1, line2, line3, line4, title

    anim = FuncAnimation(
        fig,
        update,
        frames=len(time3),
        interval=5,  # ms between frames
        blit=False,
    )

    if ap.parse_args().save_frames:
        writer = FFMpegWriter(
            fps=60, codec="libx264", bitrate=2000, extra_args=["-pix_fmt", "yuv420p"]
        )
        anim.save("temp/convergence/simulation.mp4", writer=writer)

    plt.show()


if __name__ == "__main__":
    main()
