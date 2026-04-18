from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import csv

from mpi4py import MPI
from dolfinx.io import gmshio, XDMFFile
from dolfinx import fem
from dolfinx.fem import functionspace
from dolfinx.fem import Constant, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from dolfinx import default_scalar_type
import ufl

from dolfinx.io import VTKFile

def solve_one_mesh(
    msh_path: Path,
    V_lower: float,
    V_upper: float,
    V_outer: float | None = 0.0,
    eps_r: float = 1.0,
) -> tuple[fem.Function, fem.Function, dict]:
    """
    Solve electrostatics on one air-only mesh.

    Expected facet tags from your .geo:
      - 10: force_segment (subset of upper electrode boundary)
      - 11: upper_plate (rest of upper electrode boundary)
      - 12: lower_plate
      - 20: outer boundary circle
    """

    comm = MPI.COMM_WORLD

    domain, cell_tags, facet_tags = gmshio.read_from_msh(
        str(msh_path), comm, 0, gdim=2
    )

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    # --- Function space
    V = functionspace(domain, ("Lagrange", 1))

    # --- Boundary facets by physical tag
    facets_upper = np.concatenate([facet_tags.find(10), facet_tags.find(11)])
    facets_lower = facet_tags.find(12)

    # --- Dirichlet dofs
    dofs_upper = locate_dofs_topological(V, fdim, facets_upper)
    dofs_lower = locate_dofs_topological(V, fdim, facets_lower)

    bc_upper = dirichletbc(Constant(domain, default_scalar_type(V_upper)), dofs_upper, V)
    bc_lower = dirichletbc(Constant(domain, default_scalar_type(V_lower)), dofs_lower, V)

    bcs = [bc_upper, bc_lower]

    # Optional: outer boundary Dirichlet (if you want a grounded outer circle)
    if V_outer is not None:
        facets_outer = facet_tags.find(20)
        if len(facets_outer) > 0:
            dofs_outer = locate_dofs_topological(V, fdim, facets_outer)
            bc_outer = dirichletbc(Constant(domain, default_scalar_type(V_outer)), dofs_outer, V)
            bcs.append(bc_outer)

    # --- Variational problem: -div(eps grad(phi)) = 0
    phi = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    eps = fem.Constant(domain, default_scalar_type(eps_r))  # relative permittivity
    a = eps * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx
    L = fem.Constant(domain, default_scalar_type(0.0)) * v * ufl.dx

    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    phi_h = problem.solve()
    phi_h.name = "phi"

    # --- Electric field E = -grad(phi) projected to DG0 vector field
    Vdg0 = fem.functionspace(domain, ("DG", 0, (domain.geometry.dim,)))
    E = fem.Function(Vdg0)
    E.name = "E"

    u_vec = ufl.TrialFunction(Vdg0)
    w_vec = ufl.TestFunction(Vdg0)

    a_proj = ufl.inner(u_vec, w_vec) * ufl.dx
    L_proj = ufl.inner(-ufl.grad(phi_h), w_vec) * ufl.dx

    proj = LinearProblem(
        a_proj, L_proj,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    E_h = proj.solve()
    E.x.array[:] = E_h.x.array

    # --- Diagnostics (global scalars)
    # Electrostatic energy in the air domain (up to eps0 and thickness scaling)
    energy_form = 0.5 * eps * ufl.dot(ufl.grad(phi_h), ufl.grad(phi_h)) * ufl.dx
    energy = fem.assemble_scalar(fem.form(energy_form))
    energy = comm.allreduce(energy, op=MPI.SUM)

    Vdiff = float(V_lower - V_upper)
    cap_like = (2.0 * energy / (Vdiff * Vdiff)) if abs(Vdiff) > 0 else np.nan

    info = {
        "energy": float(energy),
        "cap_like": float(cap_like),
        "V_upper": float(V_upper),
        "V_lower": float(V_lower),
        "V_outer": float(V_outer) if V_outer is not None else np.nan,
        "eps_r": float(eps_r),
    }

    return phi_h, E, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-dir", type=Path, required=True, help="Folder with .msh files (step_*.msh).")
    ap.add_argument("--pattern", type=str, default="step_*.msh", help="Glob pattern for meshes.")
    ap.add_argument("--outdir", type=Path, default=Path("electro_out"), help="Output folder for XDMF results.")
    ap.add_argument("--Vlower", type=float, default=1.0, help="Dirichlet potential on lower plate (tag 12).")
    ap.add_argument("--Vupper", type=float, default=0.0, help="Dirichlet potential on upper plate (tags 10+11).")
    ap.add_argument("--Vouter", type=float, default=0.0,
                    help="Dirichlet potential on outer boundary (tag 20). Use --no-outer-bc for Neumann.")
    ap.add_argument("--no-outer-bc", action="store_true", help="Do not apply Dirichlet on outer boundary (natural Neumann).")
    ap.add_argument("--epsr", type=float, default=1.0, help="Relative permittivity in air (default 1.0).")
    ap.add_argument("--dt", type=float, default=1.0, help="Time step used only for ParaView time series.")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank

    msh_files = sorted(args.mesh_dir.glob(args.pattern))
    if len(msh_files) == 0:
        raise FileNotFoundError(f"No meshes found: {args.mesh_dir}/{args.pattern}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for k, msh in enumerate(msh_files):
        Vouter = None if args.no_outer_bc else args.Vouter

        phi_h, E, info = solve_one_mesh(
            msh_path=msh,
            V_lower=args.Vlower,
            V_upper=args.Vupper,
            V_outer=Vouter,
            eps_r=args.epsr,
        )

        stem = msh.stem
        xdmf_path = args.outdir / f"{stem}.xdmf"

        # Write mesh + functions for ParaView
        with XDMFFile(comm, str(xdmf_path), "w") as xdmf:
            xdmf.write_mesh(phi_h.function_space.mesh)
            xdmf.write_function(phi_h)
            xdmf.write_function(E)

        if rank == 0:
            summary_rows.append({"step": k, "mesh": msh.name, **info})
            print(f"[{k+1}/{len(msh_files)}] wrote {xdmf_path.name}  energy={info['energy']:.6e} cap_like={info['cap_like']:.6e}")

    # Write summary.csv on rank 0
    if rank == 0:
        csv_path = args.outdir / "summary.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nSummary: {csv_path}")
        
    series_path = args.outdir / "electro_series.pvd"
    with VTKFile(MPI.COMM_WORLD, str(series_path), "w") as vtk:
        for k, msh in enumerate(msh_files):
            t = k * args.dt   # add --dt argument, or read from your coeff CSV

            phi_h, E, info = solve_one_mesh(
                msh_path=msh,
                V_lower=args.Vlower,
                V_upper=args.Vupper,
                V_outer=None if args.no_outer_bc else args.Vouter,
                eps_r=args.epsr,
            )

            # Important: write mesh + fields at the SAME time value
            vtk.write_mesh(phi_h.function_space.mesh, t)
            vtk.write_function(phi_h, t)
            vtk.write_function(E, t)

    if MPI.COMM_WORLD.rank == 0:
        print(f"Open in ParaView: {series_path}")


if __name__ == "__main__":
    main()
