
from pathlib import Path
import numpy as np


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
    # Compare the L2 norms of the displacement fields
    L_m = 1e-4

    # read modal_history.csv and compare the L2 norms of the displacement fields
    print("")
    
    x = np.linspace(0, L_m, 100)
    u = compute_displacement_field(2, x, L_m, "coupled_work")
    u_ref = compute_displacement_field(2, x, L_m, "temp/run_nmodes_2")
    diff = u - u_ref
    # Integrate in L2 the difference over the length of the beam (row-wise)
    # and then integrate it over time (column-wise)
    l2_norm = np.sqrt(np.trapezoid(np.trapezoid(diff**2, x), dx=1e-5))
    # Do the same for the reference solution
    l2_norm_ref = np.sqrt(np.trapezoid(np.trapezoid(u_ref**2, x), dx=1e-5))
    print("DL-ROM vs classical ROM")
    print("Comparison of displacement fields for 2 modes:")
    print(f"{'L2 norm of difference':35s} {l2_norm:.6e}")
    print(f"{'L2 norm of classical ROM':35s} {l2_norm_ref:.6e}")
    print(f"{'Relative L2 error':35s} {l2_norm / l2_norm_ref:.6e}")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
