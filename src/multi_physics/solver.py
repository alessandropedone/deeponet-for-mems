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
- ``--gmsh``: Path to the Gmsh executable (default: "gmsh").
- ``--mshver``: Gmsh mesh format version to use ("2.2" or "4.1", default: "4.1"). Note that the code currently expects physical tags to be integers, which is the case in both versions, but the internal handling of tags differs between versions. Version 4.1 is recommended for better performance and support for larger meshes.
- ``--dt``: Time step size for the Newmark integration (default: 1e-6 seconds).
- ``--nsteps``: Total number of time steps to simulate (default: 200).
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
import numpy as np

from mpi4py import MPI
from dolfinx.io import gmshio, VTKFile
from dolfinx import fem
from dolfinx.fem import functionspace, Constant, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from dolfinx import default_scalar_type
import ufl

UM = 1e-6  # micron -> meter


# ----------------------------
# Utilities
# ----------------------------
def run(cmd: list[str]) -> None:
    """
    .. admonition:: Description

        Run a subprocess command and raise an error if it fails, including the output in the error message.

    :param cmd: List of command arguments to run (e.g., ["gmsh", "-2", "mesh.geo", "-o", "mesh.msh"]).
    """
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed:\n  {' '.join(cmd)}\n\nOutput:\n{r.stdout}")


def render_geo_template(template_text: str, coeff_um: np.ndarray) -> str:
    """
    .. admonition:: Description

        Replace `__COEFF1__`, ..., `__COEFF4__` in the template text with the given coefficients (in microns).

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


def make_mesh_step(
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
    geo_text = render_geo_template(template_text, coeff_um)
    geo_path.write_text(geo_text)

    fmt = "msh2" if mshver == "2.2" else "msh4"
    run([gmsh_exec, "-2", str(geo_path), "-format", fmt, "-o", str(msh_path)])
    return msh_path


# ----------------------------
# Diagnostics helpers
# ----------------------------
def mesh_stats(domain) -> dict:
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


def tag_counts(facet_tags) -> dict:
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


def project_E_dg0(domain, phi) -> fem.Function:
    """
    .. admonition:: Description

        Project the electric field E = -grad(phi) onto a discontinuous Galerkin (DG0) function space to compute the cell-wise average electric field magnitude, which is used for diagnostics.

    :param domain: The FEniCSx mesh domain object.
    :param phi: The electrostatic potential function defined on the mesh.

    :returns:
        - Eh (``fem.Function``) -- A function in the DG0 space representing the projected electric field magnitude.
    """
    Vdg0 = fem.functionspace(domain, ("DG", 0, (domain.geometry.dim,)))
    u = ufl.TrialFunction(Vdg0)
    v = ufl.TestFunction(Vdg0)
    a = ufl.inner(u, v) * ufl.dx
    L = ufl.inner(-ufl.grad(phi), v) * ufl.dx
    prob = LinearProblem(a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    Eh = prob.solve()
    Eh.name = "E"
    return Eh


def energy_and_cap(
    domain, phi, Vdiff, eps_r=1.0, eps0=8.8541878128e-12
) -> tuple[float, float]:
    """
    .. admonition:: Description

        Compute the electrostatic energy W stored in the system and an effective capacitance :math:`C = \\frac{2W}{V_{\\text{diff}}^2}` based on the potential distribution phi and the voltage difference Vdiff.

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
    W_form = 0.5 * eps * ufl.dot(ufl.grad(phi), ufl.grad(phi)) * ufl.dx
    W = fem.assemble_scalar(fem.form(W_form))
    W = domain.comm.allreduce(W, op=MPI.SUM)
    C = (2.0 * W / (Vdiff * Vdiff)) if abs(Vdiff) > 0 else np.nan
    return float(W), float(C)


# ----------------------------
# Electrostatics
# ----------------------------
def solve_electrostatics_one(
    msh_path: Path,
    V_lower: float,
    V_upper: float = 0.0,
    V_outer: float | None = None,  # None -> natural Neumann
    eps_r: float = 1.0,
) -> tuple:
    """
    .. admonition:: Description

        Solve the electrostatic potential distribution for a given mesh and boundary conditions. The mesh is read from the specified Gmsh file, and Dirichlet boundary conditions are applied on the lower, upper, and optionally outer boundaries. The function returns the FEniCSx domain, the computed potential function, and the facet tags for further processing.

    :param msh_path: Path to the Gmsh mesh file to read.
    :param V_lower: Voltage to apply on the lower conductor (tag 12).
    :param V_upper: Voltage to apply on the upper conductor (tags 10 and 11, default: 0.0 V).
    :param V_outer: Voltage to apply on the outer boundary (tag 20). If None, natural Neumann conditions are applied instead (default: None).
    :param eps_r: Relative permittivity of the medium (default: 1.0).

    :returns:
        - domain (``mesh.Mesh``) -- The FEniCSx domain object.
        - phi (``fem.Function``) -- The computed electrostatic potential function.
        - facet_tags (``mesh.MeshTags``) -- The facet tags for further processing.
    """
    comm = MPI.COMM_WORLD
    domain, cell_tags, facet_tags = gmshio.read_from_msh(str(msh_path), comm, 0, gdim=2)

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

    eps = fem.Constant(domain, default_scalar_type(eps_r))
    a = eps * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = fem.Constant(domain, default_scalar_type(0.0)) * v * ufl.dx

    problem = LinearProblem(
        a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    phi = problem.solve()
    phi.name = "phi"
    return domain, phi, facet_tags


# ----------------------------
# Mode shapes + force projection (4 modes)
# ----------------------------
def cantilever_shape(
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


def modal_forces_4(
    domain,
    facet_tags,
    phi,
    betas: np.ndarray,  # shape (4,) in 1/m
    xmin_m: float,  # m
    L_m: float,  # m
    thickness_m: float,  # m
    eps_r: float = 1.0,
    eps0: float = 8.8541878128e-12,
) -> np.ndarray:
    """
    .. admonition:: Description

        Compute the modal forces :math:`F_i` for the first 4 modes by integrating the Maxwell traction over the force segment (tag 10) weighted by the mode shape functions. The force for each mode is given by:

        .. math::

            F_i = \\text{thickness} \\cdot \\int_{ds(10)} \\mathbf{t}_{\\text{beam}} \\cdot \\boldsymbol{\\psi}_i \\, ds, \\quad i=1..4 \\text{ with } \\boldsymbol{\\psi}_i = (0, \\text{mode}_i(x)).

        The Maxwell traction is computed from the electric field :math:`\\mathbf{E} = -\\nabla \\phi` and the permittivity, and the mode shapes are evaluated at the spatial coordinate :math:`\\xi = x - x_{\\min}` along the beam.

    :param domain: The FEniCSx mesh domain object.
    :param facet_tags: The facet tags object from the FEniCSx mesh, which contains information about the physical tags assigned to facets.
    :param phi: The electrostatic potential function defined on the mesh.
    :param betas: Array of mode parameters (beta_i) for the first 4 modes, in units of 1/m.
    :param xmin_m: The minimum x-coordinate of the beam in meters (used to compute xi).
    :param L_m: The length of the cantilever beam in meters (used in the mode shape function).
    :param thickness_m: The thickness of the beam in meters (used to scale the integrated force).
    :param eps_r: Relative permittivity of the medium (default: 1.0).
    :param eps0: Vacuum permittivity (default: 8.8541878128e-12 F/m).

    :returns:
        - F (``np.ndarray``) -- Array of modal forces for the first 4 modes.
    """
    comm = domain.comm
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    x = ufl.SpatialCoordinate(domain)
    xi = x[0] - xmin_m

    n = ufl.FacetNormal(domain)
    E = -ufl.grad(phi)
    I = ufl.Identity(domain.geometry.dim)

    eps = eps0 * eps_r
    T = eps * (ufl.outer(E, E) - 0.5 * ufl.dot(E, E) * I)
    t_beam = -ufl.dot(T, n)  # force on conductor

    F = np.zeros(4, dtype=float)
    for i in range(4):
        mode_i = cantilever_shape(xi, float(betas[i]), L_m)
        psi_i = ufl.as_vector((0.0, mode_i))
        Fi_form = thickness_m * ufl.dot(t_beam, psi_i) * ds(10)
        Fi = fem.assemble_scalar(fem.form(Fi_form))
        Fi = comm.allreduce(Fi, op=MPI.SUM)
        F[i] = float(Fi)
    return F


# ----------------------------
# Newmark (vector, diagonal modal system)
# ----------------------------
def newmark_step_diag(M, C, K, q, qd, qdd, F, dt, beta=0.25, gamma=0.5):
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
    ap.add_argument("--gmsh", type=str, default="gmsh")
    ap.add_argument("--mshver", type=str, default="4.1", choices=["2.2", "4.1"])

    ap.add_argument("--dt", type=float, default=1e-6)
    ap.add_argument("--nsteps", type=int, default=200)

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

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if shutil.which(args.gmsh) is None:
        raise RuntimeError(f"gmsh executable '{args.gmsh}' not found on PATH.")

    work_mesh = args.workdir / "meshes"
    work_out = args.workdir / "results"
    work_mesh.mkdir(parents=True, exist_ok=True)
    work_out.mkdir(parents=True, exist_ok=True)

    xmin_m = args.xmin_um * UM
    L_m = args.L_um * UM
    thickness_m = args.thickness_um * UM

    roots = np.array(
        [1.875104068711961, 4.694091132974174, 7.854757438237612, 10.995540734875466],
        dtype=float,
    )
    betas = roots / L_m  # 1/m

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
    csv_path = work_out / "modal_history.csv"

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
                "Emax_Vpm",
                "energy_J",
                "cap_like_F",
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

    with VTKFile(comm, str(vtk_path), "w") as vtk:
        for k in range(args.nsteps):
            t = k * args.dt

            # --- Remesh (rank 0) ---
            if rank == 0:
                coeff_m = q.copy()
                msh_path = make_mesh_step(
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
            Vlower = V_lower_time(t, args.Vdc, args.Vac, args.freq)
            Vouter = None if args.no_outer_bc else args.Vouter
            domain, phi, facet_tags = solve_electrostatics_one(
                msh_path=msh_path,
                V_lower=Vlower,
                V_upper=args.Vupper,
                V_outer=Vouter,
                eps_r=args.epsr,
            )

            # --- Diagnostics (mesh + tags) ---
            stats = mesh_stats(domain)
            tags = tag_counts(facet_tags)

            if args.fail_fast:
                ok = True
                if stats["nnodes"] < args.min_nodes or stats["ncells"] < args.min_cells:
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

            # --- Modal forces (4) ---
            F = modal_forces_4(
                domain=domain,
                facet_tags=facet_tags,
                phi=phi,
                betas=betas,
                xmin_m=xmin_m,
                L_m=L_m,
                thickness_m=thickness_m,
                eps_r=args.epsr,
                eps0=eps0,
            )

            # --- Field scale diagnostics ---
            phi_arr = phi.x.array
            phi_min = float(domain.comm.allreduce(phi_arr.min(), op=MPI.MIN))
            phi_max = float(domain.comm.allreduce(phi_arr.max(), op=MPI.MAX))

            # E magnitude max (DG0 projection) — component-blocked layout
            Eh = project_E_dg0(domain, phi)
            arr = Eh.x.array
            nc = domain.topology.index_map(domain.topology.dim).size_local
            Ex = arr[0:nc]
            Ey = arr[nc : 2 * nc]
            E2max_local = float(np.max(Ex * Ex + Ey * Ey)) if nc > 0 else 0.0
            E2max = float(domain.comm.allreduce(E2max_local, op=MPI.MAX))
            Emax = float(np.sqrt(E2max))

            Vdiff = float(Vlower - args.Vupper)
            W, Cap = energy_and_cap(domain, phi, Vdiff, eps_r=args.epsr, eps0=eps0)

            # --- Mechanical update (4 modes) ---
            q, qd, qdd = newmark_step_diag(M, C, K, q, qd, qdd, F, args.dt)
            q_static = np.where(K != 0, F / K, np.nan)

            # --- Write ParaView time series ---
            vtk.write_mesh(domain, t)
            vtk.write_function(phi, t)

            # --- Print diagnostics ---
            if rank == 0 and (k % args.print_every == 0):
                print(
                    f"[{k:04d}] t={t:.3e} V={Vlower:.3f}  "
                    f"q(um)=[{q[0]/UM:+.4f},{q[1]/UM:+.4f},{q[2]/UM:+.4f},{q[3]/UM:+.4f}]  "
                    f"F(N)=[{F[0]:+.2e},{F[1]:+.2e},{F[2]:+.2e},{F[3]:+.2e}]  "
                    f"F/K(um)=[{q_static[0]/UM:+.4f},{q_static[1]/UM:+.4f},{q_static[2]/UM:+.4f},{q_static[3]/UM:+.4f}]  "
                    f"phi=[{phi_min:.3f},{phi_max:.3f}] Emax={Emax:.3e} V/m  "
                    f"W={W:.3e}J C~={Cap:.3e}F  "
                    f"nodes={stats['nnodes']} cells={stats['ncells']} "
                    f"tags(10,11,12,20)=({tags['n10']},{tags['n11']},{tags['n12']},{tags['n20']})"
                )

            # --- Save CSV ---
            if rank == 0:
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
                        stats["nnodes"],
                        stats["ncells"],
                        tags["n10"],
                        tags["n11"],
                        tags["n12"],
                        tags["n20"],
                    ]
                )

    if rank == 0:
        fcsv.close()
        print(f"\nParaView time series: {vtk_path}")
        print(f"Modal history CSV:    {csv_path}")


if __name__ == "__main__":
    main()
