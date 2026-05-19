"""
Coupled multi-physics solver for electrostatic actuation of a cantilever beam (air electrostatics coupled to modal mechanics), implemented in FEniCSx with remeshing at each time step.

- Mesh template: it uses MICRONS as the unit, then mesh coordinates are converted to meters in the code for SI consistency.
- Electrostatic force: given number of mode shapes, Maxwell traction is projected onto the first cantilever mode shapes (transverse-only) on tag=10 (`force_segment`).
- Mechanics (diagonal modal system):

  .. math::

      m_i \\ddot{q}_i + c_i \\dot{q}_i + k_i q_i = F_i, \\quad i = 1 \\ldots 4,

  where :math:`k_i = m_i \\omega_i^2` and :math:`c_i = 2 \\zeta_i \\sqrt{m_i k_i}`,
  integrated with Newmark average acceleration :math:`(\\beta = 1/4,\\ \\gamma = 1/2)`.

Example usage::

    python -m coupled_modal_electro --template-geo geometry.geo --dt 1e-5 --nsteps 200 --Vdc 0 --Vac 5 --freq 2.5e3 --Vupper 0 --Vouter 0 --omega 6.3e5 3.9e6 1.1e7 2.1e7 --mass 1e-12 1e-12 1e-12 1e-12 --zeta 0.01 0.01 0.01 0.01 --print-every 1 --fail-fast

There are several optional arguments to customize the behavior:

- ``--template-geo``: Path to the Gmsh geometry template file (required). This file should contain placeholders ``__COEFF1__``, ``__COEFF2__``, ``__COEFF3__``, and ``__COEFF4__`` which will be replaced by the current modal coefficients (in microns) at each time step.
- ``--workdir``: Base directory for output files (default: "coupled_work"). Meshes will be saved in ``workdir/meshes`` and results in ``workdir/results``.
- ``--derivative-nn-path``: Optional path to a neural network model that can be used to predict the normal derivative of the potential on the force segment (tag 10) instead of computing it from the gradient. If provided, the code will call the model at each time step with appropriate input features to get the predicted dphi/dn values for use in the force computation.
- ``--no-postprocessing``: If set, the code will not run post-processing steps (i.e., saving the potential field to a ParaView file and writing the modal history CSV).
- ``--postprocessing-step``: Frequency of post-processing steps in terms of time steps (default: every step). For example, if set to 10, the code will save results and write to CSV every 10 time steps.
- ``--gmsh``: Path to the Gmsh executable (default: "gmsh").
- ``--mshver``: Gmsh mesh format version to use ("2.2" or "4.1", default: "4.1"). Note that the code currently expects physical tags to be integers, which is the case in both versions, but the internal handling of tags differs between versions. Version 4.1 is recommended for better performance and support for larger meshes.
- ``--dt``: Time step size for the Newmark integration (default: 1e-6 seconds).
- ``--nsteps``: Total number of time steps to simulate (default: 200).
- ``--nmodes``: Number of modes to compute and include in the simulation (default: 4). The current implementation supports up to 4 modes, and the mode shapes are hardcoded for a cantilever beam. If you want to use more modes or different geometries, the code would need to be generalized.
- ``--xmin-um``, ``--L-um``, ``--thickness-um``: Geometric parameters of the cantilever beam in microns (default: -50, 100, 10).
- ``--Vupper``, ``--Vouter``: Voltages for the upper conductor and outer boundary (default: 0.0 V). Use ``--no-outer-bc`` to apply natural Neumann conditions on the outer boundary instead of a Dirichlet condition.
- ``--epsr``: Relative permittivity of the medium (default: 1.0).
- ``--Vdc``, ``--Vac``, ``--freq``: Parameters for the time-varying voltage applied to the lower conductor: DC offset, AC amplitude, and frequency in Hz (default: 0 V, 20 V, 1e5 Hz).
- ``--omega``, ``--mass``, ``--zeta``: Mechanical parameters for the 4 modes: natural frequencies (rad/s), modal masses (kg), and damping ratios (default: 0.01 for all modes).
- ``--print-every``: Frequency of printing diagnostics to the console (default: every step).
- ``--fail-fast``: If set, the simulation will check for basic mesh quality and tag presence at each step and will abort if the mesh looks broken (e.g., too few nodes/cells or missing critical tags). This can help catch issues early in the remeshing process.
- ``--min-nodes``, ``--min-cells``: Thresholds for the minimum number of nodes and cells in the mesh when ``--fail-fast`` is enabled (default: 2000 each).

.. outputs:

    :file:`results/electro_series.pvd`: ParaView time series containing the electrostatic potential field at each time step.
    :file:`results/modal_history.csv`: CSV file with the time history of modal coefficients, forces, field extrema, energy, and mesh statistics for each time step. (q_i, F_i, diagnostics vs time)
"""

from __future__ import annotations

from pathlib import Path
import argparse
import shutil
import subprocess
import csv
from matplotlib.pylab import beta
from matplotlib.pylab import beta
import numpy as np
import time
import re
import sys
import os

from mpi4py import MPI
from dolfinx.io import gmshio, VTKFile
from dolfinx import fem
from dolfinx.fem import functionspace, Constant, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from dolfinx import default_scalar_type
import ufl

from src.data.fom import compute_boundary_normals_and_midpoints
from src.surrogate.losses import masked_mse, masked_mae
from src.surrogate.model import (
    DenseNetwork,
    FourierFeatures,
    LogUniformFreqInitializer,
    EinsumLayer,
    DeepONet,
)

UM = 1e-6  # micron -> meter


# ----------------------------
# Utilities
# ----------------------------
def _run(cmd: list[str]) -> None:
    """
    .. admonition:: Description

        Run a subprocess command and raise an error if it fails, including the output in the error message.

    :param cmd: List of command arguments to run (e.g., ["gmsh", "-2", "mesh.geo", "-o", "mesh.msh"]).
    """
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed:\n  {' '.join(cmd)}\n\nOutput:\n{r.stdout}")


def _render_geo_template(template_text: str, coeff_um: np.ndarray) -> str:
    """
    .. admonition:: Description

        Replace `__COEFF1__`, `__COEFF2__`, `__COEFF3__`, `__COEFF4__` in the template text with the given coefficients (in microns).

    :param template_text: The Gmsh geometry template text.
    :param coeff_um: Array of coefficients in microns.

    :returns:
        - geo_text (``str``) -- The rendered Gmsh geometry text with coefficients substituted.
    """
    c1, c2, c3, c4 = [float(x) for x in coeff_um]
    return (
        template_text.replace("__COEFF1__", f"{c1:.16g}")
        .replace("__COEFF2__", f"{c2:.16g}")
        .replace("__COEFF3__", f"{c3:.16g}")
        .replace("__COEFF4__", f"{c4:.16g}")
    )


def _make_mesh_step(
    template_geo: Path,
    workdir: Path,
    step: int,
    coeff_m: np.ndarray,  # meters, shape (4,)
    gmsh_exec: str = "gmsh",
    mshver: str = "4.1",
) -> Path:
    """
    .. admonition:: Description

        Generate a Gmsh mesh for the given time step by rendering the geometry template with the current modal coefficients and running Gmsh to produce a mesh file.

    :param template_geo: Path to the Gmsh geometry template file.
    :param workdir: Directory where the generated mesh file will be saved.
    :param step: Current time step index (used for naming the output files).
    :param coeff_m: Array of modal coefficients in meters (shape (4,)) to substitute into the geometry template.
    :param gmsh_exec: Name or path of the Gmsh executable to run (default: "gmsh").
    :param mshver: Gmsh mesh format version to use ("2.2" or "4.1", default: "4.1").

    :returns:
        - msh_path (``Path``) -- Path to the generated Gmsh mesh file for this time step.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    tag = f"{step:05d}"
    geo_path = workdir / f"step_{tag}.geo"
    msh_path = workdir / f"step_{tag}.msh"

    template_text = template_geo.read_text()
    coeff_um = coeff_m / UM
    geo_text = _render_geo_template(template_text, coeff_um)
    geo_path.write_text(geo_text)

    fmt = "msh2" if mshver == "2.2" else "msh4"
    _run([gmsh_exec, "-2", str(geo_path), "-format", fmt, "-o", str(msh_path)])
    return msh_path


# ----------------------------
# Diagnostics helpers
# ----------------------------
def _mesh_stats(domain) -> dict:
    """
    .. admonition:: Description

        Compute basic statistics about the mesh, including the number of nodes, number of cells, and bounding box coordinates.

    :param domain: The FEniCSx mesh domain object.

    :returns:
        - stats (``dict``) -- A dictionary containing mesh statistics: number of nodes, number of cells, and bounding box coordinates.
    """
    x = domain.geometry.x
    bbox_min = x.min(axis=0)
    bbox_max = x.max(axis=0)
    nn = domain.geometry.x.shape[0]
    nc = domain.topology.index_map(domain.topology.dim).size_local
    return {
        "nnodes": int(nn),
        "ncells": int(nc),
        "xmin": float(bbox_min[0]),
        "xmax": float(bbox_max[0]),
        "ymin": float(bbox_min[1]),
        "ymax": float(bbox_max[1]),
    }


def _tag_counts(facet_tags) -> dict:
    """
    .. admonition:: Description

        Count the number of facets for each relevant tag (10, 11, 12, 20) in the facet_tags object.

    :param facet_tags: The facet tags object from the FEniCSx mesh, which contains information about the physical tags assigned to facets.

    :returns:
        - tags (``dict``) -- A dictionary containing the count of facets for each tag of interest (10, 11, 12, 20).
    """

    def n(tag: int) -> int:
        try:
            return int(len(facet_tags.find(tag)))
        except Exception:
            return 0

    return {"n10": n(10), "n11": n(11), "n12": n(12), "n20": n(20)}


def _project_E_dg0(domain, phi) -> fem.Function:
    """
    .. admonition:: Description

        Project the electric field E = -grad(phi) onto a discontinuous Galerkin (DG0) function space to compute the cell-wise average electric field magnitude, which is used for diagnostics.

    :param domain: The FEniCSx mesh domain object.
    :param phi: The electrostatic potential function defined on the mesh.

    :returns:
        - Eh (``fem.Function``) -- A function in the DG0 space representing the projected electric field magnitude.
    """
    # Define the vector function space for the gradient
    domain.geometry.x[:] /= UM
    Vdg0 = fem.functionspace(domain, ("DG", 0, (domain.geometry.dim,)))
    # Define the trial and test functions for the vector space
    u = ufl.TrialFunction(Vdg0)
    v = ufl.TestFunction(Vdg0)
    # Define the gradient of the solution
    E = -ufl.grad(phi)
    # Define the bilinear and linear forms
    a = ufl.inner(u, v) * ufl.dx
    L = ufl.inner(E, v) * ufl.dx
    # Assemble the system
    problem = LinearProblem(
        a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    Eh = problem.solve()
    Eh.name = "E"
    Eh.x.array[:] /= UM
    return Eh


def _energy_and_cap(
    domain, phi, Vdiff, eps_r=1.0, eps0=8.8541878128e-12
) -> tuple[float, float]:
    """
    .. admonition:: Description

        Compute the electrostatic energy `W` stored in the system and an effective capacitance `C=2W/Vdiff^2` based on the potential distribution `phi` and the voltage difference `Vdiff`.

    :param domain: The FEniCSx mesh domain object.
    :param phi: The electrostatic potential function defined on the mesh.
    :param Vdiff: The voltage difference between the conductors :math:`(V_{\\text{lower}} - V_{\\text{upper}})`.
    :param eps_r: Relative permittivity of the medium (default: 1.0).
    :param eps0: Vacuum permittivity (default: 8.8541878128e-12 F/m).

    :returns:
        - W (``float``) -- The computed electrostatic energy in joules.
        - C (``float``) -- The computed effective capacitance in farads.
    """
    eps = eps0 * eps_r
    # Linear dielectric
    W_form = 0.5 * eps * ufl.dot(ufl.grad(phi), ufl.grad(phi)) * ufl.dx
    W = fem.assemble_scalar(fem.form(W_form))
    W = domain.comm.allreduce(W, op=MPI.SUM)
    C = (2.0 * W / (Vdiff * Vdiff)) if abs(Vdiff) > 0 else np.nan
    return float(W), float(C)


# ----------------------------
# Electrostatics
# ----------------------------
def _solve_electrostatics_one(
    msh_path: Path,
    V_lower: float,
    V_upper: float = 0.0,
    V_outer: float | None = None,  # None -> natural Neumann
) -> tuple:
    """
    .. admonition:: Description

        Solve the electrostatic potential distribution for a given mesh and boundary conditions. The mesh is read from the specified Gmsh file, and Dirichlet boundary conditions are applied on the lower, upper, and optionally outer boundaries. The function returns the FEniCSx domain, the computed potential function, and the facet tags for further processing.

    :param msh_path: Path to the Gmsh mesh file to read.
    :param V_lower: Voltage to apply on the lower conductor (tag 12).
    :param V_upper: Voltage to apply on the upper conductor (tags 10 and 11, default: 0.0 V).
    :param V_outer: Voltage to apply on the outer boundary (tag 20). If None, natural Neumann conditions are applied instead (default: None).

    :returns:
        - domain (``mesh.Mesh``) -- The FEniCSx domain object.
        - phi (``fem.Function``) -- The computed electrostatic potential function.
        - facet_tags (``mesh.MeshTags``) -- The facet tags for further processing.
    """
    comm = MPI.COMM_WORLD
    stdout_fd = sys.stdout.fileno()
    saved_stdout = os.dup(stdout_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stdout_fd)
    os.close(devnull)
    domain, cell_tags, facet_tags = gmshio.read_from_msh(str(msh_path), comm, 0, gdim=2)

    # Restore stdout for the rest of the function (e.g., to print diagnostics)
    os.dup2(saved_stdout, stdout_fd)

    # Convert geometry coordinates from microns to meters (SI)
    domain.geometry.x[:] *= UM

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    V = functionspace(domain, ("Lagrange", 1))

    facets_upper = np.concatenate([facet_tags.find(10), facet_tags.find(11)])
    facets_lower = facet_tags.find(12)

    dofs_upper = locate_dofs_topological(V, fdim, facets_upper)
    dofs_lower = locate_dofs_topological(V, fdim, facets_lower)

    bc_upper = dirichletbc(
        Constant(domain, default_scalar_type(V_upper)), dofs_upper, V
    )
    bc_lower = dirichletbc(
        Constant(domain, default_scalar_type(V_lower)), dofs_lower, V
    )
    bcs = [bc_upper, bc_lower]

    if V_outer is not None:
        facets_outer = facet_tags.find(20)
        if len(facets_outer) > 0:
            dofs_outer = locate_dofs_topological(V, fdim, facets_outer)
            bc_outer = dirichletbc(
                Constant(domain, default_scalar_type(V_outer)), dofs_outer, V
            )
            bcs.append(bc_outer)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = fem.Constant(domain, default_scalar_type(0.0)) * v * ufl.dx

    problem = LinearProblem(
        a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    phi = problem.solve()
    phi.name = "phi"
    return domain, phi, facet_tags


# ------------------------------
# Mode shapes
# ------------------------------
def _cantilever_shape_ufl(
    xi: ufl.core.expr.Expr, beta: float, L: float
) -> ufl.core.expr.Expr:
    """
    .. admonition:: Description

        Compute the cantilever mode shape function.
        The mode shape is given by

        .. math::

            \\psi_i(\\xi) = \\cosh(\\beta_i \\xi) - \\cos(\\beta_i \\xi) - C_i \\left( \\sinh(\\beta_i \\xi) - \\sin(\\beta_i \\xi) \\right)

        where

        .. math::

            C_i = \\frac{\\cosh(\\beta_i L) + \\cos(\\beta_i L)}{\\sinh(\\beta_i L) + \\sin(\\beta_i L)}.

    :param xi: The spatial coordinate along the beam, :math:`\\xi = x - x_{\\min}`.
    :param beta: The mode parameter :math:`\\beta_i` for the *i*-th mode, related to the natural frequency.
    :param L: The cantilever beam length.

    :returns:
        - psi (``ufl.core.expr.Expr``) -- The computed mode shape function evaluated at xi.
    """
    C = (ufl.cosh(beta * L) + ufl.cos(beta * L)) / (
        ufl.sinh(beta * L) + ufl.sin(beta * L)
    )
    return (
        ufl.cosh(beta * xi)
        - ufl.cos(beta * xi)
        - C * (ufl.sinh(beta * xi) - ufl.sin(beta * xi))
    )


def _cantilever_shape_np(xi: np.ndarray, beta: float, L: float) -> np.ndarray:
    """
    .. admonition:: Description

        Compute the cantilever mode shape function.
        The mode shape is given by

        .. math::

            \\psi_i(\\xi) = \\cosh(\\beta_i \\xi) - \\cos(\\beta_i \\xi) - C_i \\left( \\sinh(\\beta_i \\xi) - \\sin(\\beta_i \\xi) \\right)

        where

        .. math::

            C_i = \\frac{\\cosh(\\beta_i L) + \\cos(\\beta_i L)}{\\sinh(\\beta_i L) + \\sin(\\beta_i L)}.

    :param xi: The spatial coordinate along the beam, :math:`\\xi = x - x_{\\min}`.
    :param beta: The mode parameter :math:`\\beta_i` for the *i*-th mode, related to the natural frequency.
    :param L: The cantilever beam length.

    :returns:
        - psi (``np.ndarray``) -- The computed mode shape function evaluated at xi.
    """
    C = (np.cosh(beta * L) + np.cos(beta * L)) / (np.sinh(beta * L) + np.sin(beta * L))
    return (
        np.cosh(beta * xi)
        - np.cos(beta * xi)
        - C * (np.sinh(beta * xi) - np.sin(beta * xi))
    )


def _clamped_shape_ufl(
    xi: ufl.core.expr.Expr, beta: float, L: float
) -> ufl.core.expr.Expr:
    """
    .. admonition:: Description

        Compute the clamped-clamped mode shape function.
        The mode shape is given by

        .. math::

            \\psi_i(\\xi) = \\sinh(\\beta_i \\xi) - \\sin(\\beta_i \\xi) + C_i \\left( \\cosh(\\beta_i \\xi) - \\cos(\\beta_i \\xi) \\right)

        .. math::

            C_i = \\frac{\\cosh(\\beta_i L) - \\cos(\\beta_i L)}{\\sinh(\\beta_i L) + \\sin(\\beta_i L)}.

    :param xi: The spatial coordinate along the beam, :math:`\\xi = x - x_{\\min}`.
    :param beta: The mode parameter :math:`\\beta_i` for the *i*-th mode, related to the natural frequency.
    :param L: The clamped-clamped beam length.

    :returns:
        - psi (``ufl.core.expr.Expr``) -- The computed mode shape function evaluated at xi.
    """
    C = (ufl.cos(beta * L) - ufl.cosh(beta * L)) / (
        ufl.sinh(beta * L) + ufl.sin(beta * L)
    )
    return (
        ufl.sinh(beta * xi)
        - ufl.sin(beta * xi)
        + C * (ufl.cosh(beta * xi) - ufl.cos(beta * xi))
    )


def _clamped_shape_np(xi: np.ndarray, beta: float, L: float) -> np.ndarray:
    """
    .. admonition:: Description

        Compute the clamped-clamped mode shape function.
        The mode shape is given by

        .. math::

            \\psi_i(\\xi) = \\sinh(\\beta_i \\xi) - \\sin(\\beta_i \\xi) + C_i \\left( \\cosh(\\beta_i \\xi) - \\cos(\\beta_i \\xi) \\right)

        .. math::

            C_i = \\frac{\\cosh(\\beta_i L) - \\cos(\\beta_i L)}{\\sinh(\\beta_i L) + \\sin(\\beta_i L)}.

    :param xi: The spatial coordinate along the beam, :math:`\\xi = x - x_{\\min}`.
    :param beta: The mode parameter :math:`\\beta_i` for the *i*-th mode, related to the natural frequency.
    :param L: The clamped-clamped beam length.

    :returns:
        - psi (``np.ndarray``) -- The computed mode shape function evaluated at xi.
    """
    C = (np.cos(beta * L) - np.cosh(beta * L)) / (np.sinh(beta * L) + np.sin(beta * L))
    return (
        np.sinh(beta * xi)
        - np.sin(beta * xi)
        + C * (np.cosh(beta * xi) - np.cos(beta * xi))
    )


def _compute_displacement(
    x: np.ndarray, L: float, q: np.ndarray, clamped: bool = False
) -> np.ndarray:
    """
    .. admonition:: Description

        Compute the displacement field along the beam by superposing the first 4 modes weighted by their modal coefficients. The mode shapes are evaluated at the spatial coordinates `x` along the beam. The mode parameters `beta_i` are computed from the known roots of the cantilever beam characteristic equation divided by the beam length `L_m`.

    :param x: The spatial coordinates along the beam where the displacement is evaluated (1D array).
    :param L: The length of the cantilever beam (used to compute the mode parameters).
    :param q: The modal coefficients for the first `nmodes` modes (array of shape (nmodes,)).
    :param clamped: A boolean indicating whether the beam is clamped at both ends.

    :returns:
        - u (``np.ndarray``) -- The computed displacement field along the beam at the coordinates `x`, resulting from the superposition of the first `nmodes` mode shapes weighted by their coefficients `q`.
    """

    if not clamped:
        roots = np.array(
            [
                1.875104068711961,
                4.694091132974174,
                7.854757438237612,
                10.995540734875466,
            ],
            dtype=float,
        )
        betas = roots / L
        modes = np.array([_cantilever_shape_np(x, betas[i], L) for i in range(4)])
    else:
        roots = np.array(
            [4.73004074486270, 7.85320462409584, 10.9956078380017, 14.1371654912575],
            dtype=float,
        )
        betas = roots / L
        modes = np.array([_clamped_shape_np(x, betas[i], L) for i in range(4)])
    u = q @ modes
    return u


def compute_normals_ordered(points: np.ndarray) -> np.ndarray:
    """
    .. admonition:: Description

        Compute the outward normal vectors at a set of ordered points along the lower edge (e.g., the boundary of the beam) by computing the tangent vector from neighboring points. The function assumes that the points are ordered along the curve.

    :param points: An array of shape (n, 2) containing the coordinates of the points along the curve in order.


    :returns:
        - normals (``np.ndarray``) -- An array of shape (n-1, 2) containing the estimated normal vectors at each midpoint, normalized to unit length and oriented outward.
    """
    npts = len(points)

    # Tangents
    tangents = np.empty((npts - 1, 2), dtype=float)
    # t = p_{i+1} - p_i
    tangents[:, 0] = points[1:, 0] - points[:-1, 0]
    tangents[:, 1] = points[1:, 1] - points[:-1, 1]

    # Rotate by +90°
    normals = np.empty_like(tangents)
    normals[:, 0] = -tangents[:, 1]
    normals[:, 1] = tangents[:, 0]

    # Force downward orientation
    flip = normals[:, 1] > 0
    normals[flip] *= -1.0

    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= norms

    return normals


# ------------------------------
# Force projection
# ------------------------------
def _modal_forces_4(
    nmodes: int,
    betas: np.ndarray,
    xmin_m: float,
    L_m: float,
    thickness_m: float,
    dphidn: tuple[np.ndarray, np.ndarray, np.ndarray] = None,
    phi: tuple[fem.Function, ufl.Domain, ufl.Measure] = None,
    eps_r: float = 1.0,
    eps0: float = 8.8541878128e-12,
    clamped: bool = False,
) -> np.ndarray:
    """
    .. admonition:: Description

        Compute the modal forces :math:`F_i` for the first :math:`n\\leq 4` modes by integrating the Maxwell traction over the force segment (tag 10) weighted by the mode shape functions. The force for each mode is given by:

        .. math::

            F_i = \\text{thickness} \\cdot \\int_{ds(10)} \\mathbf{t}_{\\text{beam}} \\cdot \\boldsymbol{\\psi}_i \\, ds, \\quad i=1..n \\text{ with } \\boldsymbol{\\psi}_i = (0, \\text{mode}_i(x)).

        The Maxwell traction is computed from the electric field :math:`\\mathbf{E} = -\\nabla \\phi` and the permittivity, and the mode shapes are evaluated at the spatial coordinate :math:`\\xi = x - x_{\\min}` along the beam.

    :param nmodes: The number of modes to compute forces for (up to 4).
    :param betas: Array of mode parameters :math:`\\beta_i` for the first 4 modes, related to the natural frequencies.
    :param xmin_m: The minimum x coordinate of the beam (used to compute the spatial coordinate xi).
    :param L_m: The length of the cantilever beam (used to compute the mode shapes).
    :param thickness_m: The thickness of the beam in the out-of-plane direction (used to scale the force).
    :param dphidn: Optional tuple containing the normal derivative of the potential on the plate segment, the corresponding coordinates of the midpoints and the integration points used to compute the force without FEniCSx forms. If provided, it should be a tuple of (dphidn_values, midpoints, integration_points) where dphidn_values is an array of the normal derivative values at the midpoints, midpoints is an array of the (x,y) coordinates of the midpoints, and integration_points is an array of the integration point coordinates. Both midpoints and integration_points are assumed to be increasing in x.
    :param phi: Optional tuple containing the potential function, domain, and facet tags, used to compute the force using FEniCSx forms. If provided, it should be a tuple of (phi_function, domain, facet_tags) where phi_function is the computed potential function, domain is the FEniCSx mesh domain, and facet_tags contains the physical tags for the facets.
    :param eps_r: Relative permittivity of the medium (default: 1.0).
    :param eps0: Vacuum permittivity (default: 8.8541878128e-12 F/m).
    :param clamped: A boolean indicating whether the beam is clamped at both ends (default: False). This affects the mode shapes used in the force computation.

    :returns:
        - F (``np.ndarray``) -- Array of modal forces for the first 4 modes.

    :raises ValueError:
        - If `nmodes` is greater than 4, since the function is designed for up to 4 modes and would require generalization for more modes.
        - If neither `phi` nor `dphidn` is provided, since at least one of them is needed to compute the forces.
    """
    if nmodes > 4:
        raise ValueError(
            "This function is designed for up to 4 modes. For more modes, a generalization is needed."
        )

    eps = eps0 * eps_r

    if phi is not None:
        phi, domain, facet_tags = phi
        n = ufl.FacetNormal(domain)
        if dphidn is not None:
            dphidn_vals, midpoints, integration_points = dphidn
            Q = fem.functionspace(domain, ("DG", 0))
            dphidn = fem.Function(Q)
            # Facets on tag 10
            boundary_facets = facet_tags.find(10)
            if len(dphidn_vals) != len(boundary_facets):
                raise ValueError(
                    "Output size of the network does not match number of facets on tag 10."
                )
            fdim = domain.topology.dim - 1
            tdim = domain.topology.dim
            domain.topology.create_connectivity(fdim, tdim)
            facet_to_cell = domain.topology.connectivity(fdim, tdim)
            boundary_cells = np.unique(
                np.hstack([facet_to_cell.links(f) for f in boundary_facets])
            )
            dphidn.x.array[boundary_cells] = dphidn_vals
        else:
            dphidn = ufl.dot(ufl.grad(phi), n)
        comm = domain.comm
        ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
        x = ufl.SpatialCoordinate(domain)
        xi = x[0] - xmin_m
        t_beam = -0.5 * eps * dphidn**2 * n
        # Since we only need the normal component for the force on the beam,  we can use the normal derivative directly as shown above.
        # The commented-out code below shows the more general approach using the Maxwell stress tensor.
        # I = ufl.Identity(domain.geometry.dim)
        # T = eps * (ufl.outer(E, E) - 0.5 * ufl.dot(E, E) * I)
        # t_beam = -ufl.dot(T, n)  # force on conductor
        F = np.zeros(4, dtype=float)
        for i in range(nmodes):
            if clamped:
                mode_i = _clamped_shape_ufl(xi, float(betas[i]), L_m)
            else:
                mode_i = _cantilever_shape_ufl(xi, float(betas[i]), L_m)
            psi_i = ufl.as_vector((0.0, mode_i))
            Fi_form = thickness_m * ufl.dot(t_beam, psi_i) * ds(10)
            Fi = fem.assemble_scalar(fem.form(Fi_form))
            Fi = comm.allreduce(Fi, op=MPI.SUM)
            F[i] = float(Fi)
    else:
        if dphidn is None:
            raise ValueError(
                "Either phi or dphidn must be provided to compute the forces."
            )
        # Extract the normal derivative values and midpoints from the provided dphidn tuple
        dphidn_vals, midpoints, integration_points = dphidn
        dx = np.linalg.norm(integration_points[1:] - integration_points[:-1], axis=1)
        # Compute the force without using FEniCSx forms, using the provided normal derivative values directly on the midpoints
        normals = compute_normals_ordered(integration_points)
        t_beam_vals = 0.5 * eps * dphidn_vals**2 * normals[:, 1]
        F = np.zeros(4, dtype=float)
        for i in range(nmodes):
            xi = midpoints[:, 0] - xmin_m
            if clamped:
                mode_i = _clamped_shape_np(xi, float(betas[i]), L_m)
            else:
                mode_i = _cantilever_shape_np(xi, float(betas[i]), L_m)
            Fi_local = thickness_m * np.sum(t_beam_vals * mode_i * dx)
            Fi = MPI.COMM_WORLD.allreduce(Fi_local, op=MPI.SUM)
            F[i] = float(Fi)
    return F


# ----------------------------
# Newmark (vector, diagonal modal system)
# ----------------------------
def _newmark_step_diag(
    M: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    F: np.ndarray,
    dt: float,
    beta: float = 0.25,
    gamma: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    .. admonition:: Description

            Perform one time step of the Newmark average acceleration method for a diagonal modal system, where M, C, K are arrays of modal masses, damping coefficients, and stiffnesses for each mode. The function updates the modal coordinates q, velocities qd, and accelerations qdd based on the applied forces F and the time step dt.
            In particular, we have

            .. math:: M qdd + C qd + K q = F \quad \text{for each mode},

            and the Newmark update equations are applied in a vectorized manner for all modes simultaneously.

    :param M: Array of modal masses for each mode.
    :param C: Array of modal damping coefficients for each mode.
    :param K: Array of modal stiffnesses for each mode.
    :param q: Array of modal coordinates at the current time step.
    :param qd: Array of modal velocities at the current time step.
    :param qdd: Array of modal accelerations at the current time step.
    :param F: Array of modal forces applied at the current time step.
    :param dt: Time step size for the integration.
    :param beta: Newmark beta parameter (default: 0.25 for average acceleration).
    :param gamma: Newmark gamma parameter (default: 0.5 for average acceleration).

    :returns:
        - q_new (``np.ndarray``) -- Updated modal coordinates after the time step.
        - qd_new (``np.ndarray``) -- Updated modal velocities after the time step.
        - qdd_new (``np.ndarray``) -- Updated modal accelerations after the time step.
    """
    q_pred = q + dt * qd + 0.5 * dt * dt * (1.0 - 2.0 * beta) * qdd
    qd_pred = qd + dt * (1.0 - gamma) * qdd

    denom = M + gamma * dt * C + beta * dt * dt * K
    qdd_new = (F - C * qd_pred - K * q_pred) / denom

    q_new = q_pred + beta * dt * dt * qdd_new
    qd_new = qd_pred + gamma * dt * qdd_new
    return q_new, qd_new, qdd_new


# ----------------------------
# Voltage law
# ----------------------------
def V_lower_time(t: float, Vdc: float, Vac: float, freq_hz: float) -> float:
    """
    .. admonition:: Description

            Compute the lower voltage as a function of time.

    :param t: Time (s).
    :param Vdc: DC voltage (V).
    :param Vac: AC voltage amplitude (V).
    :param freq_hz: Frequency (Hz).

    :returns:
        - V_lower (``float``) -- Lower voltage at time t (V).
    """
    return Vdc + Vac * np.sin(2.0 * np.pi * freq_hz * t)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template-geo", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, default=Path("coupled_work"))
    ap.add_argument(
        "--derivative-nn-path",
        type=Path,
        help="Path to the neural network for the derivative.",
        default=None,
    )
    ap.add_argument(
        "--potential-nn-path",
        type=Path,
        help="Path to the neural network for the potential.",
        default=None,
    )
    ap.add_argument(
        "--no-postprocessing",
        action="store_true",
        help="If set to True, it will also save the results in VTK format for visualization in ParaView.",
        default=False,
    )
    ap.add_argument(
        "--postprocessing-step",
        type=int,
        default=1,
        help="Interval of steps to save VTK files when --not-postprocessing is not set. Ignored if --no-postprocessing is set.",
    )
    ap.add_argument(
        "--clamped",
        action="store_true",
        help="If set, it will use clamped-clamped mode shapes instead of cantilever mode shapes.",
        default=False,
    )

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if ap.parse_known_args()[0].workdir.exists() and rank == 0:
        response = input(
            f"Working directory '{ap.parse_known_args()[0].workdir}' already exists. Do you want to delete it and continue? [y/N] "
        )
        if response.lower() != "y":
            print("Aborting.")
            return
        shutil.rmtree(ap.parse_known_args()[0].workdir)

    ap.add_argument("--gmsh", type=str, default="gmsh")
    ap.add_argument("--mshver", type=str, default="4.1", choices=["2.2", "4.1"])

    ap.add_argument("--dt", type=float, default=1e-6)
    ap.add_argument("--nsteps", type=int, default=200)
    ap.add_argument("--nmodes", type=int, default=4)
    if ap.parse_known_args()[0].nmodes > 4 and rank == 0:
        ap.error(
            "The current implementation supports up to 4 modes. Please set --nmodes to 4 or less."
        )

    ap.add_argument("--xmin-um", type=float, default=-50.0)
    ap.add_argument("--L-um", type=float, default=100.0)
    ap.add_argument("--thickness-um", type=float, default=10.0)

    ap.add_argument("--Vupper", type=float, default=0.0)
    ap.add_argument("--Vouter", type=float, default=0.0)
    ap.add_argument("--no-outer-bc", action="store_true")
    ap.add_argument("--epsr", type=float, default=1.0)

    ap.add_argument("--Vdc", type=float, default=0.0)
    ap.add_argument("--Vac", type=float, default=20.0)
    ap.add_argument("--freq", type=float, default=1e5)

    # Mechanics (4 modes)
    ap.add_argument(
        "--omega",
        type=float,
        nargs=4,
        required=True,
        help="4 natural frequencies (rad/s).",
    )
    ap.add_argument(
        "--mass", type=float, nargs=4, required=True, help="4 modal masses (kg)."
    )
    ap.add_argument(
        "--zeta",
        type=float,
        nargs=4,
        default=[0.01, 0.01, 0.01, 0.01],
        help="4 damping ratios (-).",
    )

    # Diagnostics controls
    ap.add_argument("--print-every", type=int, default=1)
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--min-nodes", type=int, default=2000)
    ap.add_argument("--min-cells", type=int, default=2000)
    args = ap.parse_args()

    if args.derivative_nn_path is not None:
        import tensorflow as tf

    if args.no_postprocessing and args.derivative_nn_path is None:
        print(
            "Warning: Postprocessing automatically activated since no surrogate is used for the derivative, which means the FEniCSx solve will be performed at every step to compute the forces. Setting --no-postprocessing to False."
        )
        args.no_postprocessing = False

    if args.derivative_nn_path is None and args.postprocessing_step > 1:
        print(
            "Warning: --postprocessing-step > 1 has no effect when no surrogate is used for the derivative, since the FEniCSx solve will be performed at every step. Setting --postprocess-step to 1."
        )
        args.postprocessing_step = 1

    if shutil.which(args.gmsh) is None:
        raise RuntimeError(f"gmsh executable '{args.gmsh}' not found on PATH.")

    # Read from the geometry template overetch and distance
    with open(args.template_geo, "r") as f:
        template_geo_text = f.read()

    def get_variable(text, var_name):
        pattern = rf"{var_name}\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*;"
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"Variable '{var_name}' not found")
        return float(match.group(1))

    overetch_um = get_variable(template_geo_text, "overetch")
    distance_um = get_variable(template_geo_text, "distance")
    overetch_m = overetch_um * UM
    distance_m = distance_um * UM
    ref_factor = int(get_variable(template_geo_text, "r"))
    n_nodes = 50 * ref_factor
    n_segments = n_nodes - 1

    work_mesh = args.workdir / "meshes"
    work_out = args.workdir / "results"
    work_mesh.mkdir(parents=True, exist_ok=True)
    work_out.mkdir(parents=True, exist_ok=True)

    xmin_m = args.xmin_um * UM
    L_m = args.L_um * UM
    xmax_m = xmin_m + L_m
    thickness_m = args.thickness_um * UM

    if args.clamped:
        roots = np.array(
            [4.73004074486270, 7.85320462409584, 10.9956078380017, 14.1371654912575],
            dtype=float,
        )
    else:
        roots = np.array(
            [
                1.875104068711961,
                4.694091132974174,
                7.854757438237612,
                10.995540734875466,
            ],
            dtype=float,
        )
    betas = roots / L_m

    omega = np.array(args.omega, dtype=float)
    M = np.array(args.mass, dtype=float)
    zeta = np.array(args.zeta, dtype=float)

    K = M * omega**2
    C = 2.0 * zeta * omega * M

    # State vectors (meters)
    q = np.zeros(4, dtype=float)
    qd = np.zeros(4, dtype=float)
    qdd = np.zeros(4, dtype=float)

    vtk_path = work_out / "electro_series.pvd"
    vtk_nn_path = work_out / "electro_series_nn.pvd"
    vtk_error_path = work_out / "electro_series_error.pvd"
    csv_path = args.workdir / "modal_history.csv"
    execution_time_path = args.workdir / "execution_time.csv"

    if rank == 0:
        fcsv = csv_path.open("w", newline="")
        writer = csv.writer(fcsv)
        writer.writerow(
            [
                "step",
                "t_s",
                "Vlower_V",
                "q1_m",
                "q2_m",
                "q3_m",
                "q4_m",
                "qd1_mps",
                "qd2_mps",
                "qd3_mps",
                "qd4_mps",
                "qdd1_mps2",
                "qdd2_mps2",
                "qdd3_mps2",
                "qdd4_mps2",
                "F1_N",
                "F2_N",
                "F3_N",
                "F4_N",
                "phi_min_V",
                "phi_max_V",
                "|E|_max_Vpm",
                "energy_J",
                "cap_like_F",
                "cap_like_F_approx",
                "nnodes",
                "ncells",
                "n10",
                "n11",
                "n12",
                "n20",
            ]
        )
    else:
        fcsv = None
        writer = None

    eps0 = 8.8541878128e-12

    if args.derivative_nn_path is not None:
        stdout_fd = sys.stdout.fileno()
        saved_stdout = os.dup(stdout_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stdout_fd)
        os.close(devnull)
        dphidn_nn = tf.keras.models.load_model(
            "{}".format(args.derivative_nn_path),
            custom_objects={
                "DenseNetwork": DenseNetwork,
                "FourierFeatures": FourierFeatures,
                "LogUniformFreqInitializer": LogUniformFreqInitializer,
                "EinsumLayer": EinsumLayer,
                "DeepONet": DeepONet,
                "masked_mse": masked_mse,
                "masked_mae": masked_mae,
            },
        )
        os.dup2(saved_stdout, stdout_fd)
        print("\033[38;2;0;175;6m\n\nLoaded derivative surrogate.\033[0m")

    if args.potential_nn_path is not None:
        stdout_fd = sys.stdout.fileno()
        saved_stdout = os.dup(stdout_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stdout_fd)
        os.close(devnull)
        phi_nn = tf.keras.models.load_model(
            "{}".format(args.potential_nn_path),
            custom_objects={
                "DenseNetwork": DenseNetwork,
                "FourierFeatures": FourierFeatures,
                "LogUniformFreqInitializer": LogUniformFreqInitializer,
                "EinsumLayer": EinsumLayer,
                "DeepONet": DeepONet,
                "masked_mse": masked_mse,
                "masked_mae": masked_mae,
            },
        )
        os.dup2(saved_stdout, stdout_fd)
        print("\033[38;2;0;175;6m\n\nLoaded potential surrogate.\033[0m")

    start = time.perf_counter()
    postproc_time = 0
    solution_time = 0
    F = np.zeros(4, dtype=float)
    correction = 0.0

    with VTKFile(comm, str(vtk_path), "w") as vtk:
        for k in range(args.nsteps):
            t = k * args.dt
            Vlower = V_lower_time(t, args.Vdc, args.Vac, args.freq)

            sol = time.perf_counter()
            # --- Mechanical update ---
            q, qd, qdd = _newmark_step_diag(M, C, K, q, qd, qdd, F, args.dt)
            q_static = np.where(K != 0, F / K, np.nan)
            solution_time = solution_time + time.perf_counter() - sol

            postproc = time.perf_counter()
            if (
                not args.no_postprocessing and (k % args.postprocessing_step == 0)
            ) or k == 1:
                # --- Remesh (rank 0) ---
                if rank == 0:
                    coeff_m = q.copy()
                    print("")
                    print("-" * 65)
                    print(
                        f"Meshing with q = [{coeff_m[0]:.2e}, {coeff_m[1]:.2e}, {coeff_m[2]:.2e}, {coeff_m[3]:.2e}]"
                    )
                    print("-" * 65)
                    print("")
                    msh_path = _make_mesh_step(
                        template_geo=args.template_geo,
                        workdir=work_mesh,
                        step=k,
                        coeff_m=coeff_m,
                        gmsh_exec=args.gmsh,
                        mshver=args.mshver,
                    )
                else:
                    msh_path = None

                comm.barrier()
                if rank != 0:
                    msh_path = work_mesh / f"step_{k:05d}.msh"

                # --- Electrostatics ---
                Vouter = None if args.no_outer_bc else args.Vouter
                domain, phi, facet_tags = _solve_electrostatics_one(
                    msh_path=msh_path,
                    V_lower=Vlower,
                    V_upper=args.Vupper,
                    V_outer=Vouter,
                )

                # --- Diagnostics (mesh + tags) ---
                stats = _mesh_stats(domain)
                tags = _tag_counts(facet_tags)

                if args.fail_fast:
                    ok = True
                    if (
                        stats["nnodes"] < args.min_nodes
                        or stats["ncells"] < args.min_cells
                    ):
                        ok = False
                    if tags["n10"] == 0 or tags["n12"] == 0:
                        ok = False
                    if not ok:
                        if rank == 0:
                            print(f"\nFAIL-FAST at step {k}: mesh/tags look broken")
                            print(f"  nodes={stats['nnodes']} cells={stats['ncells']}")
                            print(
                                f"  n10={tags['n10']} n11={tags['n11']} n12={tags['n12']} n20={tags['n20']}"
                            )
                            print(
                                f"  bbox x[{stats['xmin']/UM:.3f},{stats['xmax']/UM:.3f}] um"
                                f" y[{stats['ymin']/UM:.3f},{stats['ymax']/UM:.3f}] um"
                            )
                        break
            postproc_time = postproc_time + time.perf_counter() - postproc

            # --- Modal forces ---
            if args.derivative_nn_path is not None:
                sol = time.perf_counter()
                # Compute midpoints directly from the geometry information
                x_nodes_m = np.linspace(xmin_m, xmax_m, n_nodes)
                y_nodes_m = (
                    distance_m / 2
                    + overetch_m
                    + _compute_displacement(
                        x_nodes_m - xmin_m, L_m, q, clamped=args.clamped
                    )
                )
                x_mid_m = 0.5 * (x_nodes_m[:-1] + x_nodes_m[1:])
                y_mid_m = np.full(
                    n_segments,
                    distance_m / 2
                    + overetch_m
                    + _compute_displacement(
                        x_mid_m - xmin_m, L_m, q, clamped=args.clamped
                    ),
                )
                midpoints_m = np.column_stack((x_mid_m, y_mid_m))
                nodes_m = np.column_stack((x_nodes_m, y_nodes_m))
                midpoints_um = midpoints_m / UM
                F = _modal_forces_4(
                    nmodes=args.nmodes,
                    betas=betas,
                    xmin_m=xmin_m,
                    L_m=L_m,
                    thickness_m=thickness_m,
                    dphidn=[
                        Vlower
                        * dphidn_nn(
                            [
                                np.concatenate(
                                    [np.array([overetch_um, distance_um]), q / UM]
                                ),
                                midpoints_um[np.newaxis, :, :],
                            ]
                        )
                        .numpy()
                        .squeeze()
                        / UM,
                        midpoints_m,
                        nodes_m,
                    ],
                    eps_r=args.epsr,
                    eps0=eps0,
                    clamped=args.clamped,
                )
                # Estimate the capacitance using the displacement
                solution_time = solution_time + time.perf_counter() - sol
                postproc = time.perf_counter()
                if (
                    not args.no_postprocessing and (k % args.postprocessing_step == 0)
                ) or k == 1:
                    # --- Field scale diagnostics ---
                    phi_arr = phi.x.array
                    phi_min = float(domain.comm.allreduce(phi_arr.min(), op=MPI.MIN))
                    phi_max = float(domain.comm.allreduce(phi_arr.max(), op=MPI.MAX))

                    # E magnitude max (DG0 projection) — component-blocked layout
                    Eh = _project_E_dg0(domain, phi)
                    dim = domain.geometry.dim
                    Ex = Eh.x.array[0::dim]
                    Ey = Eh.x.array[1::dim]
                    E2_local_max = np.max(Ex**2 + Ey**2)
                    E2max = domain.comm.allreduce(E2_local_max, op=MPI.MAX)
                    Emax = float(np.sqrt(E2max))
                    Vdiff = float(Vlower - args.Vupper)
                    W, Cap = _energy_and_cap(
                        domain, phi, Vdiff, eps_r=args.epsr, eps0=eps0
                    )

                    # --- Write ParaView time series ---
                    vtk.write_mesh(domain, t)
                    vtk.write_function(phi, t)

                    # Construct surrogate predictions for potential and error fields for visualization in ParaView
                    if args.potential_nn_path is not None:
                        # Extract x and y coordinates
                        V = phi.function_space
                        dofs_uh = np.arange(V.dofmap.index_map.size_local)
                        dofs_c = V.tabulate_dof_coordinates()[dofs_uh]
                        x_um = np.array(dofs_c[:, :2], dtype=float)
                        phi_pred_arr = (
                            Vlower
                            * phi_nn(
                                [
                                    np.concatenate(
                                        [np.array([overetch_um, distance_um]), q / UM]
                                    ),
                                    x_um[np.newaxis, :, :],
                                ]
                            )
                            .numpy()
                            .squeeze()
                        )
                        # Save the prediction
                        phi_pred = fem.Function(phi.function_space)
                        phi_pred.name = "phi_pred"
                        phi_pred.x.array[:] = phi_pred_arr
                        phi_pred.x.scatter_forward()
                        vtk.write_function(phi_pred, t)
                        # Save the error
                        phi_error = fem.Function(phi.function_space)
                        phi_error.name = "phi_error"
                        phi_error.x.array[:] = abs(phi.x.array - phi_pred_arr)
                        phi_error.x.scatter_forward()
                        vtk.write_function(phi_error, t)
                else:
                    stats = {
                        "nnodes": np.nan,
                        "ncells": np.nan,
                        "xmin": np.nan,
                        "xmax": np.nan,
                        "ymin": np.nan,
                        "ymax": np.nan,
                    }
                    tags = {"n10": np.nan, "n11": np.nan, "n12": np.nan, "n20": np.nan}
                    phi_min = np.nan
                    phi_max = np.nan
                    Emax = np.nan
                    W = np.nan
                    Cap = np.nan
                postproc_time = postproc_time + time.perf_counter() - postproc
            else:
                sol = time.perf_counter()
                normals, midpoints_m = compute_boundary_normals_and_midpoints(
                    domain, facet_tags.find(10)
                )
                midpoints_um = midpoints_m / UM
                F = _modal_forces_4(
                    nmodes=args.nmodes,
                    betas=betas,
                    xmin_m=xmin_m,
                    L_m=L_m,
                    thickness_m=thickness_m,
                    phi=[phi, domain, facet_tags],
                    eps_r=args.epsr,
                    eps0=eps0,
                    clamped=args.clamped,
                )
                solution_time = solution_time + time.perf_counter() - sol

                postproc = time.perf_counter()
                # --- Field scale diagnostics ---
                phi_arr = phi.x.array
                phi_min = float(domain.comm.allreduce(phi_arr.min(), op=MPI.MIN))
                phi_max = float(domain.comm.allreduce(phi_arr.max(), op=MPI.MAX))

                # E magnitude max (DG0 projection) — component-blocked layout
                Eh = _project_E_dg0(domain, phi)
                dim = domain.geometry.dim
                Ex = Eh.x.array[0::dim]
                Ey = Eh.x.array[1::dim]
                E2_local_max = np.max(Ex**2 + Ey**2)
                E2max = domain.comm.allreduce(E2_local_max, op=MPI.MAX)
                Emax = float(np.sqrt(E2max))

                Vdiff = float(Vlower - args.Vupper)
                W, Cap = _energy_and_cap(domain, phi, Vdiff, eps_r=args.epsr, eps0=eps0)

                # --- Write ParaView time series ---
                vtk.write_mesh(domain, t)
                vtk.write_function(phi, t)

                if args.potential_nn_path is not None:
                    # Extract x and y coordinates
                    V = phi.function_space
                    dofs_uh = np.arange(V.dofmap.index_map.size_local)
                    dofs_c = V.tabulate_dof_coordinates()[dofs_uh]
                    x_um = np.array(dofs_c[:, :2], dtype=float)
                    phi_pred_arr = (
                        Vlower
                        * phi_nn(
                            [
                                np.concatenate(
                                    [np.array([overetch_um, distance_um]), q / UM]
                                ),
                                x_um[np.newaxis, :, :],
                            ]
                        )
                        .numpy()
                        .squeeze()
                    )
                    # Save the prediction
                    phi_pred = fem.Function(phi.function_space)
                    phi_pred.name = "phi_pred"
                    phi_pred.x.array[:] = phi_pred_arr
                    phi_pred.x.scatter_forward()
                    vtk.write_function(phi_pred, t)
                    # Save the error
                    phi_error = fem.Function(phi.function_space)
                    phi_error.name = "phi_error"
                    phi_error.x.array[:] = abs(phi.x.array - phi_pred_arr)
                    phi_error.x.scatter_forward()
                    vtk.write_function(phi_error, t)

                postproc_time = postproc_time + time.perf_counter() - postproc

            postproc = time.perf_counter()
            # --- Capacitance approximation ---
            x = np.linspace(0, L_m, n_segments)
            if args.clamped:
                modes = np.array(
                    [_clamped_shape_np(x, betas[i], L_m) for i in range(4)]
                )
            else:
                modes = np.array(
                    [_cantilever_shape_np(x, betas[i], L_m) for i in range(4)]
                )
            u = q @ modes
            Cap_approx = eps0 * np.trapezoid((1 / (distance_m + u)), x)
            if (
                not args.no_postprocessing and (k % args.postprocessing_step == 0)
            ) or k == 1:
                correction = (
                    Cap - Cap_approx
                    if not np.isnan(Cap) and not np.isnan(Cap_approx)
                    else 0.0
                )
            Cap_approx += correction
            postproc_time = postproc_time + time.perf_counter() - postproc

            # --- Print diagnostics ---
            if rank == 0 and (k % args.print_every == 0):
                # Print nan if postprocessing is disabled or not performed at this step
                print(
                    f"\n{'='*90}"
                    f"\nStep: {k:04d}, "
                    f"Time: {t:.3e} s"
                    f"\n{'-'*90}"
                    f"\nVlower = {Vlower:.3f} V  "
                    f"Phi_min = {phi_min:.3f} V  "
                    f"Phi_max = {phi_max:.3f} V  "
                    f"|E|_max = {Emax:.3e} V/m "
                    f"\nEnergy = {W:.3e} J  "
                    f"Cap = {Cap:.3e} F  "
                    f"Cap_approx = {Cap_approx:.3e} F  "
                    f"\nNodes = {stats['nnodes']} "
                    f"Cells = {stats['ncells']}  "
                    f"\n{'-'*90}"
                    f"\nTags   : "
                    f"n10={tags['n10']},  "
                    f"n11={tags['n11']},  "
                    f"n12={tags['n12']},  "
                    f"n20={tags['n20']}"
                    f"\n{'-'*90}"
                    f"\n Mode   |    q (um)    |    F (N)     |   F/K (um)"
                    f"\n        |              |              |           "
                    f"\n   1    | {q[0]/UM:+12.2e} | {F[0]:+12.2e} | {q_static[0]/UM:+10.4f}"
                    f"\n   2    | {q[1]/UM:+12.2e} | {F[1]:+12.2e} | {q_static[1]/UM:+10.4f}"
                    f"\n   3    | {q[2]/UM:+12.2e} | {F[2]:+12.2e} | {q_static[2]/UM:+10.4f}"
                    f"\n   4    | {q[3]/UM:+12.2e} | {F[3]:+12.2e} | {q_static[3]/UM:+10.4f}"
                    f"\n{'='*90}\n"
                )

            postproc = time.perf_counter()
            # --- Save CSV ---
            if rank == 0 and k % args.print_every == 0:
                writer.writerow(
                    [
                        k,
                        t,
                        Vlower,
                        q[0],
                        q[1],
                        q[2],
                        q[3],
                        qd[0],
                        qd[1],
                        qd[2],
                        qd[3],
                        qdd[0],
                        qdd[1],
                        qdd[2],
                        qdd[3],
                        F[0],
                        F[1],
                        F[2],
                        F[3],
                        phi_min,
                        phi_max,
                        Emax,
                        W,
                        Cap,
                        Cap_approx,
                        stats["nnodes"],
                        stats["ncells"],
                        tags["n10"],
                        tags["n11"],
                        tags["n12"],
                        tags["n20"],
                    ]
                )
            postproc_time = postproc_time + time.perf_counter() - postproc

    end = time.perf_counter()
    total_time = end - start
    fcsv = execution_time_path.open("w", newline="")
    writer = csv.writer(fcsv)
    writer.writerow(["total_s", "postprocessing_and_meshing_s", "solution_s"])
    writer.writerow([total_time, postproc_time, solution_time])

    if rank == 0:
        fcsv.close()
        print("")
        print("=" * 90)
        print(f"ParaView time series:         {vtk_path}")
        print(f"ParaView time series (NN):    {vtk_nn_path}")
        print(f"ParaView time series (Error): {vtk_error_path}")
        print(f"Modal history CSV:            {csv_path}")
        print(f"Execution time CSV:           {execution_time_path}")
        print(f"Total runtime: {total_time:.2f} seconds")
        print(f"  Postprocessing/meshing time: {postproc_time:.2f} seconds")
        print(f"  Solution time:               {solution_time:.2f} seconds")
        print("=" * 90)


if __name__ == "__main__":
    main()
