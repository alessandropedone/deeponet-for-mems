from pathlib import Path
import numpy as np


def cantilever_shape_np(xi: np.ndarray, beta: float, L: float) -> np.ndarray:
    """
    Compute the cantilever mode shape function in NumPy for post-processing and comparison with the modal history.
    """
    C = (np.cosh(beta * L) + np.cos(beta * L)) / (np.sinh(beta * L) + np.sin(beta * L))
    return (
        np.cosh(beta * xi)
        - np.cos(beta * xi)
        - C * (np.sinh(beta * xi) - np.sin(beta * xi))
    )


def compute_displacement_from_history(
    nmodes: int, x: np.ndarray, L_m: float, path: Path
) -> np.ndarray:
    """
    Compute the displacement field from the modal history CSV file.
    """
    roots = np.array(
        [1.875104068711961, 4.694091132974174, 7.854757438237612, 10.995540734875466],
        dtype=float,
    )
    betas = roots / L_m
    modes = np.array([cantilever_shape_np(x, betas[i], L_m) for i in range(nmodes)])
    csv_path = path / "modal_history.csv"
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    q_values = data[:, 3 : 3 + nmodes]
    u = q_values @ modes
    return u
