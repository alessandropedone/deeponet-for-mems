#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.io import gmshio
from dolfinx.fem import dirichletbc
from dolfinx.fem.petsc import LinearProblem
from dolfinx import default_scalar_type


# -----------------------------
# Gmsh I/O
# -----------------------------
def write_geo_from_template(template_path: str, out_geo: str, coeff_um: np.ndarray) -> None:
    coeff_um = np.asarray(coeff_um, dtype=float)
    if coeff_um.shape != (4,):
        raise ValueError("coeff_um must have shape (4,)")

    if not np.isfinite(coeff_um).all():
        raise RuntimeError(f"Non-finite coeff_um: {coeff_um}")

    with open(template_path, "r") as f:
        s = f.read()

    s = s.replace("__COEFF1__", f"{coeff_um[0]:.16e}")
    s = s.replace("__COEFF2__", f"{coeff_um[1]:.16e}")
    s = s.replace("__COEFF3__", f"{coeff_um[2]:.16e}")
    s = s.replace("__COEFF4__", f"{coeff_um[3]:.16e}")

    with open(out_geo, "w") as f:
        f.write(s)


def gmsh_mesh(geo_path: str, msh_path: str, step: int) -> None:
    cmd = ["gmsh", "-2", geo_path, "-format", "msh41", "-o", msh_path, "-rand", "0", "-v", "2"]
    p = subprocess.run(cmd, text=True, capture_output=True)

    if p.returncode != 0:
        fail_geo = geo_path.replace(".geo", f"_FAIL_step{step:06d}.geo")
        shutil.copyfile(geo_path, fail_geo)

        msg = "\n".join([
            f"Gmsh failed at step {step}. Saved: {fail_geo}",
            "----- gmsh stdout (tail) -----",
            p.stdout[-4000:],
            "----- gmsh stderr (tail) -----",
            p.stderr[-4000:],
        ])
        raise RuntimeError(msg)


# -----------------------------
# Tag transfer: parent -> submesh
# -----------------------------
def facet_tag_submesh(parent, parent_facet_tags, sub, sub_vertex_map, tags_of_interest):
    tdim = parent.topology.dim
    fdim = tdim - 1
    parent.topology.create_connectivity(fdim, 0)
    sub.topology.create_connectivity(fdim, 0)

    parent_fv = parent.topology.connectivity(fdim, 0)
    sub_fv = sub.topology.connectivity(fdim, 0)

    key_to_subfacet = {}
    n_sf = sub.topology.index_map(fdim).size_local
    for sf in range(n_sf):
        vs_sub = sub_fv.links(sf)
        pv = np.sort(sub_vertex_map[vs_sub])
        key_to_subfacet[(int(pv[0]), int(pv[1]))] = sf

    sub_facets = []
    sub_values = []
    for tag in tags_of_interest:
        pfacets = parent_facet_tags.find(tag)
        for pf in pfacets:
            vs_parent = parent_fv.links(int(pf))
            pv = np.sort(vs_parent)
            key = (int(pv[0]), int(pv[1]))
            if key in key_to_subfacet:
                sub_facets.append(key_to_subfacet[key])
                sub_values.append(int(tag))

    fdim = sub.topology.dim - 1
    if len(sub_facets) == 0:
        return dmesh.meshtags(sub, fdim, np.array([], dtype=np.int32), np.array([], dtype=np.int32))
    return dmesh.meshtags(sub, fdim, np.array(sub_facets, dtype=np.int32), np.array(sub_values, dtype=np.int32))


# -----------------------------
# Electrostatics (air domain)
# -----------------------------
def solve_electrostatics(air_mesh, air_facet_tags, V_upper: float, V_lower: float) -> fem.Function:
    Vphi = fem.functionspace(air_mesh, ("CG", 1))
    phi = ufl.TrialFunction(Vphi)
    q = ufl.TestFunction(Vphi)

    a = ufl.inner(ufl.grad(phi), ufl.grad(q)) * ufl.dx
    L = fem.Constant(air_mesh, default_scalar_type(0.0)) * q * ufl.dx

    fdim = air_mesh.topology.dim - 1

    def dofs_on_tag(tag: int) -> np.ndarray:
        facets = air_facet_tags.find(tag)
        if len(facets) == 0:
            return np.array([], dtype=np.int32)
        return fem.locate_dofs_topological(Vphi, fdim, facets)

    dofs_upper = np.concatenate([dofs_on_tag(t) for t in [10, 11, 40, 41]]).astype(np.int32)
    dofs_lower = dofs_on_tag(12)
    dofs_outer = dofs_on_tag(20)

    dofs_all = np.unique(np.concatenate([dofs_upper, dofs_lower, dofs_outer])).astype(np.int32)
    if dofs_all.size == 0:
        raise RuntimeError("No Dirichlet dofs found for electrostatics (tags missing).")

    g = fem.Function(Vphi)
    g.x.array[:] = 0.0
    g.x.array[dofs_outer] = 0.0
    g.x.array[dofs_lower] = float(V_lower)
    g.x.array[dofs_upper] = float(V_upper)

    bc = dirichletbc(g, dofs_all)

    opts = {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-10,
        "ksp_max_it": 2000,
        "ksp_error_if_not_converged": None,
    }

    phih = LinearProblem(a, L, bcs=[bc], petsc_options=opts).solve()
    phih.name = "phi"
    if not np.isfinite(phih.x.array).all():
        raise RuntimeError("Electrostatics returned NaN/Inf in solution vector.")
    return phih


# -----------------------------
# Beam ROM
# -----------------------------
@dataclass
class BeamParams:
    L: float       # length (m)
    h: float       # in-plane height (m) (4 um)
    b: float       # out-of-plane thickness (m) (30 um)
    E: float
    rho: float
    zeta: float    # modal damping ratio (same for all modes)


def cantilever_shape_numpy(x: np.ndarray, beta: float, L: float) -> np.ndarray:
    C = (np.cosh(beta * L) + np.cos(beta * L)) / (np.sinh(beta * L) + np.sin(beta * L))
    return np.cosh(beta * x) - np.cos(beta * x) - C * (np.sinh(beta * x) - np.sin(beta * x))


def build_modal_mk(params: BeamParams, Nq: int, n_int: int = 4001):
    roots = np.array([1.875104068711961, 4.694091132974174, 7.854757438237612, 10.995540734875466], dtype=float)
    beta = roots[:Nq] / params.L

    x = np.linspace(0.0, params.L, n_int)
    A = params.b * params.h
    I = params.b * params.h**3 / 12.0

    Phi = np.zeros((Nq, x.size), dtype=float)
    m = np.zeros(Nq, dtype=float)
    for i in range(Nq):
        Phi[i] = cantilever_shape_numpy(x, beta[i], params.L)
        m[i] = params.rho * A * np.trapz(Phi[i] ** 2, x)

    omega = (beta**2) * np.sqrt(params.E * I / (params.rho * A))
    k = m * omega**2
    c = 2.0 * params.zeta * omega * m
    return beta, x, Phi, m, c, k, omega


def enforce_min_gap_um(coeff_um: np.ndarray, distance_um: float, gap_min_um: float) -> np.ndarray:
    """
    Scales coeff_um (uniform scaling) so that min gap >= gap_min_um.
    Gap(x) = (distance/2 + defl(x)) - (-distance/2) = distance + defl(x)
    where defl(x) is the modal bottom displacement in microns.
    """
    coeff_um = coeff_um.copy()
    if np.allclose(coeff_um, 0.0):
        return coeff_um

    # sample deflection along x in microns on [0,L]
    L_um = 100.0
    x = np.linspace(0.0, L_um, 300)

    roots = np.array([1.875104068711961, 4.694091132974174, 7.854757438237612, 10.995540734875466], dtype=float)
    beta = roots / L_um

    def defl(scale: float) -> np.ndarray:
        y = np.zeros_like(x)
        for j in range(4):
            bj = beta[j]
            C = (np.cosh(bj * L_um) + np.cos(bj * L_um)) / (np.sinh(bj * L_um) + np.sin(bj * L_um))
            y += (scale * coeff_um[j]) * (np.cosh(bj * x) - np.cos(bj * x) - C * (np.sinh(bj * x) - np.sin(bj * x)))
        return y

    def min_gap(scale: float) -> float:
        return float(np.min(distance_um + defl(scale)))

    if min_gap(1.0) >= gap_min_um:
        return coeff_um

    # bisection for scale in [0,1]
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if min_gap(mid) >= gap_min_um:
            lo = mid
        else:
            hi = mid
    return coeff_um * lo


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    here = os.path.dirname(__file__)
    template_geo = os.path.join(here, "actuator_template.geo")
    mesh_dir = os.path.join(here, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    eps0 = 8.854e-12

    # Voltages
    V_upper = 0.0
    f_drive = 2500.0
    def V_lower(t: float) -> float:
        return 1.0 * np.sin(2.0 * np.pi * f_drive * t)

    # Beam parameters (SI)
    beam = BeamParams(L=100e-6, h=4e-6, b=30e-6, E=170e9, rho=2330.0, zeta=0.02)

    # ROM
    Nq = 1
    beta, x_grid, Phi_np, m, c, k, omega = build_modal_mk(beam, Nq)

    # Time
    dt = 1e-6     # start conservative
    T = 2e-4
    nsteps = int(np.floor(T / dt))

    # State (physical meters)
    q = np.zeros(Nq, dtype=float)
    v = np.zeros(Nq, dtype=float)

    # Tags
    tags_list = [10, 11, 12, 20, 40, 41]

    # Minimum gap safeguard (microns)
    gap_min_um = 0.3
    distance_um = 1.5

    if rank == 0:
        print(f"Run: dt={dt:g}, steps={nsteps}, zeta={beam.zeta}", flush=True)

    for step in range(nsteps + 1):
        t = step * dt

        if not (np.isfinite(q).all() and np.isfinite(v).all()):
            raise RuntimeError(f"Non-finite modal state at step {step}: q={q}, v={v}")

        # --- coefficients in microns for geometry ---
        coeff_um = (q * 1e6).astype(float)
        coeff_um = np.nan_to_num(coeff_um, nan=0.0, posinf=0.0, neginf=0.0)

        # enforce gap constraint (prevents invalid geometry / near-touching)
        coeff_um = enforce_min_gap_um(coeff_um, distance_um=distance_um, gap_min_um=gap_min_um)

        # write and mesh
        geo_path = os.path.join(mesh_dir, "actuator_step.geo")
        msh_path = os.path.join(mesh_dir, "actuator_step.msh")

        if rank == 0:
            write_geo_from_template(template_geo, geo_path, coeff_um)
            gmsh_mesh(geo_path, msh_path, step)
        comm.barrier()

        # read mesh (all ranks)
        parent, cell_tags, facet_tags = gmshio.read_from_msh(msh_path, comm, 0, gdim=2)
        parent.geometry.x[:] *= 1e-6  # microns -> meters

        tdim = parent.topology.dim
        fdim = tdim - 1

        air_cells = cell_tags.find(30)
        if len(air_cells) == 0:
            raise RuntimeError("No air cells with physical tag 30 (check .geo).")

        air_mesh, _, air_vmap, _ = dmesh.create_submesh(parent, tdim, air_cells)
        air_mesh.topology.create_connectivity(fdim, tdim)
        air_mesh.topology.create_connectivity(fdim, 0)

        air_facet_tags = facet_tag_submesh(parent, facet_tags, air_mesh, air_vmap, tags_list)

        if step == 0 and rank == 0:
            print("Facet tag counts (air):", flush=True)
            for tag in tags_list:
                print(f"  tag {tag}: {len(air_facet_tags.find(tag))}", flush=True)

        # electrostatics
        phih = solve_electrostatics(air_mesh, air_facet_tags, V_upper=V_upper, V_lower=V_lower(t))

        # generalized forces by boundary integral (no DG projection, no facet->cell sampling)
        ds = ufl.Measure("ds", domain=air_mesh, subdomain_data=air_facet_tags)
        x = ufl.SpatialCoordinate(air_mesh)

        # local coordinate along beam: x_local in [0, L]
        x_local = x[0] + 0.5 * beam.L

        p = 0.5 * eps0 * ufl.inner(ufl.grad(phih), ufl.grad(phih))  # Pa

        Q = np.zeros(Nq, dtype=float)

        for i in range(Nq):
            bi = float(beta[i])
            Ci = (np.cosh(bi * beam.L) + np.cos(bi * beam.L)) / (np.sinh(bi * beam.L) + np.sin(bi * beam.L))
            Phi_i = ufl.cosh(bi * x_local) - ufl.cos(bi * x_local) - Ci * (ufl.sinh(bi * x_local) - ufl.sin(bi * x_local))

            Qi_local = fem.assemble_scalar(fem.form(-beam.b * p * Phi_i * ds(10)))  # Newton
            Q[i] = comm.allreduce(Qi_local, op=MPI.SUM)

        if not np.isfinite(Q).all():
            raise RuntimeError(f"Non-finite generalized force at step {step}: Q={Q}")

        # modal ODE: m qdd + c qd + k q = Q
        a = (Q - c * v - k * q) / m

        # symplectic Euler (forward Euler variant stable for oscillators)
        v = v + dt * a
        q = q + dt * v

        if rank == 0 and step % 20 == 0:
            print(f"step {step:6d} t={t: .3e}  V={V_lower(t): .3e}  q1={q[0]: .3e} m  |Q|={np.linalg.norm(Q):.3e} N",
                  flush=True)

    if rank == 0:
        print("Done.", flush=True)


if __name__ == "__main__":
    main()
