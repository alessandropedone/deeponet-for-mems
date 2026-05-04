"""Test the effect of the number of modes on the solution."""

import subprocess
from pathlib import Path
import numpy as np
import argparse


def cantilever_shape_np(xi, beta, L) -> np.ndarray:
    """Compute the shape of the cantilever beam for a given beta and length."""
    C = (np.cosh(beta * L) + np.cos(beta * L)) / (np.sinh(beta * L) + np.sin(beta * L))
    return (
        np.cosh(beta * xi)
        - np.cos(beta * xi)
        - C * (np.sinh(beta * xi) - np.sin(beta * xi))
    )


def compute_displacement_field(nmodes, x, L_m) -> np.ndarray:
    """Compute the displacement field from the modal history CSV file."""
    roots = np.array(
        [1.875104068711961, 4.694091132974174, 7.854757438237612, 10.995540734875466],
        dtype=float,
    )
    betas = roots / L_m
    modes = np.array([cantilever_shape_np(x, betas[i], L_m) for i in range(nmodes)])
    workdir = Path(f"temp/run_nmodes_{nmodes}")
    csv_path = workdir / "modal_history.csv"
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    q_values = data[:, 3 : 3 + nmodes]
    u = q_values @ modes
    return u


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
        x = np.linspace(0, L_m, 100)
        u = compute_displacement_field(nmodes, x, L_m)
        u_ref = compute_displacement_field(1, x, L_m)  # Use 1 mode as reference
        diff = u - u_ref
        # Integrate in L2 the difference over the length of the beam (row-wise)
        # and then integrate it over time (column-wise)
        l2_norm = np.sqrt(np.trapezoid(np.trapezoid(diff**2, x), dx=1e-5))
        # Do the same for the reference solution
        l2_norm_ref = np.sqrt(np.trapezoid(np.trapezoid(u_ref**2, x), dx=1e-5))
        print(f"{'Number of modes':35s} {nmodes}")
        print(f"{'L2 norm of difference':35s} {l2_norm:.6e}")
        print(f"{'L2 norm of reference (1 mode)':35s} {l2_norm_ref:.6e}")
        print(f"{'Relative L2 error':35s} {l2_norm / l2_norm_ref:.6e}")
        print(f"{'-'*60}")


if __name__ == "__main__":
    main()
