from pathlib import Path
import alphashape
from shapely import Point
import numpy as np

from mpi4py import MPI

from dolfinx.io import gmshio
from dolfinx import fem
from dolfinx.fem import functionspace
from dolfinx.fem import (Constant, dirichletbc, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx import default_scalar_type
import ufl

import os, sys
import h5py


def _get_domain_from_mesh(mesh_path: str):
    """
    .. admonition:: Description
        
        Read the mesh from the .msh file and return the computational domain.

    :param mesh_path: Path to the mesh file.

    :returns:
        - **domain** (``fem.Domain``) -- The computational domain containing mesh information.
    """
    from mpi4py import MPI
    from dolfinx.io import gmshio

    domain, cell_tags, facet_tags = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, 0, gdim=2)

    return domain


def _compute_boundary_normals_and_midpoints(domain, boundary_facets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    .. admonition:: Description
        
        Compute boundary normals and midpoints for the facets of the upper plate (tags 10 and 11).

    :param domain: The computational domain containing mesh information.
    :param boundary_facets: The boundary facets associated with the domain.

    :returns:
        - **normals** (``np.ndarray``) -- Normal vectors at the boundary facets.
        - **midpoints** (``np.ndarray``) -- Midpoints of the boundary facets.
    """
    # 2D: facet dimension
    fdim = domain.topology.dim - 1

    # Connectivity: facets -> vertices
    facet_vertices = domain.topology.connectivity(fdim, 0)

    midpoints = []
    normals = []

    for f in boundary_facets:
        vertices = facet_vertices.links(f)  # get vertex indices for this facet
        p0 = domain.geometry.x[vertices[0]]
        p1 = domain.geometry.x[vertices[1]]
        # Compute 2D edge vector
        edge = p1 - p0
        # Compute normal vector (perpendicular to edge)
        n_vec = np.array([-edge[1], edge[0]])
        n_vec /= np.linalg.norm(n_vec)
        normals.append(n_vec)
        midpoints.append((p0 + p1)/2)

    normals = np.array(normals)
    midpoints = np.array(midpoints)[:, :2]  # only x and y coordinates
    
    # Create polygon and point objects
    polygon = alphashape.alphashape(midpoints, alpha=0.1)  # tune alpha
    eps = 1e-6
    p = np.zeros_like(midpoints)
    for n, mp in zip(normals, midpoints):
        p = mp + eps * n  
        point = Point(p) 
        if polygon.contains(point):
            n *= -1

    return normals, midpoints


def fom(mesh: str, bc_lower_plate: float = 1.0, bc_upper_plate: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    .. admonition:: Description

        Full order model that solves the PDE with given boundary conditions and computes the solutions and its gradient. 
    
    :param mesh: Path to the mesh file.
    :param bc_lower_plate: Dirichlet BC value for the lower plate.
    :param bc_upper_plate: Dirichlet BC value for the upper plate.
    
    :returns:
        - **x** (``np.ndarray``) -- x coordinates of all mesh nodes.
        - **y** (``np.ndarray``) -- y coordinates of all mesh nodes.
        - **cells** (``np.ndarray``) -- Cell connectivity information.
        - **potential** (``np.ndarray``) -- Potential values at all mesh nodes.
        - **grad_x** (``np.ndarray``) -- x component of the gradient at all mesh nodes.
        - **grad_y** (``np.ndarray``) -- y component of the gradient at all mesh nodes.
        - **midpoints_plate** (``np.ndarray``) -- Midpoints of the facets of the upper plate boundary.
        - **normal_derivatives_plate** (``np.ndarray``) -- Normal derivative values at the upper plate boundary midpoints.
        - **normal_vectors_plate** (``np.ndarray``) -- Normal vectors at the upper plate boundary midpoints.

    .. note::

        We return the midpoints of the facets of the upper plate boundary since the gradient is constant on each cell (we chose DG0).
    """

    if mesh.is_file() and mesh.suffix == ".msh":
        # Silence Gmsh chatter
        stdout_fd = sys.stdout.fileno()
        saved_stdout = os.dup(stdout_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stdout_fd)
        os.close(devnull)
        
        domain, cell_tags, facet_tags = gmshio.read_from_msh(mesh, MPI.COMM_WORLD, 0, gdim=2)
        
        # Restore stdout
        os.dup2(saved_stdout, stdout_fd)
        os.close(saved_stdout)
        
        ## SOLVE FOR THE POTENTIAL DISTRIBUTION ##

        # Define finite element function space
        V = functionspace(domain, ("Lagrange", 1))

        # Identify the boundary (create facet to cell connectivity required to determine boundary facets)
        tdim = domain.topology.dim
        fdim = tdim - 1
        domain.topology.create_connectivity(fdim, tdim)

        # Find facets marked with 10, 11, 12 (the two plates)
        facets_rect1 = np.concatenate([facet_tags.find(10), facet_tags.find(11)])
        facets_rect2 = facet_tags.find(12)

        # Locate degrees of freedom
        dofs_rect1 = locate_dofs_topological(V, fdim, facets_rect1)
        dofs_rect2 = locate_dofs_topological(V, fdim, facets_rect2)

        # Define different Dirichlet values
        u_rect1 = Constant(domain, bc_upper_plate)
        u_rect2 = Constant(domain, bc_lower_plate)

        # Create BCs
        bc1 = dirichletbc(u_rect1, dofs_rect1, V)
        bc2 = dirichletbc(u_rect2, dofs_rect2, V)

        bcs = [bc1, bc2]

        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # Source term
        f = fem.Constant(domain, default_scalar_type(0.0))

        # Variational problem
        a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = f * v * ufl.dx

        # Assemble the system
         
        problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()


        ## COMPUTE THE GRADIENT OF THE SOLUTION ##

        # Define the vector function space for the gradient
        V_grad = fem.functionspace(domain, ("DG", 0, (domain.geometry.dim, )))

        # Define the trial and test functions for the vector space
        u_vec = ufl.TrialFunction(V_grad)
        v_vec = ufl.TestFunction(V_grad)

        # Define the gradient of the solution
        grad_u = ufl.grad(uh)

        # Define the bilinear and linear forms
        a_grad = ufl.inner(u_vec, v_vec) * ufl.dx
        L_grad = ufl.inner(grad_u, v_vec) * ufl.dx

        # Assemble the system
        problem_grad = LinearProblem(a_grad, L_grad, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        grad_uh = problem_grad.solve()


        ## EXTRACT AND RETURN RELEVANT DATA ##

        # Find all dofs in the function spaces
        dofs_uh = np.arange(V.dofmap.index_map.size_local)

        # Find the dofs of the upper plate
        boundary_facets = facets_rect1
        dofs1011 = locate_dofs_topological(V, fdim, boundary_facets)

        # Extract all x and y coordinates
        dofs_c = V.tabulate_dof_coordinates()[dofs_uh]
        x = np.array(dofs_c[:, 0])
        y = np.array(dofs_c[:, 1])

        # Extract the potential values
        potential = np.array(uh.x.array[dofs_uh])

        # Extract gradient components
        dim = domain.geometry.dim
        grad_x = grad_uh.x.array[0::dim]
        grad_y = grad_uh.x.array[1::dim]

        # Extract gradient components on the upper plate
        boundary_cells = []
        for f in boundary_facets:
            # get the cell connected to this facet
            connected_cell = domain.topology.connectivity(fdim, tdim).links(f)[0]
            boundary_cells.append(connected_cell)
        grad_x_plate = grad_x[boundary_cells]
        grad_y_plate = grad_y[boundary_cells]
        
        # Extract cell connectivity
        cells = domain.topology.connectivity(domain.topology.dim, 0).array.reshape(-1, domain.geometry.dim + 1)

        # Plot potential distribution and gradient components
        # from plot_solutions import plot
        # plot(x, y, cells, potential, title="Potential Distribution", colorbar_label="Potential")
        # plot(x, y, cells, grad_x, title="Gradient X Component", colorbar_label="Gradient X", sharp_color_range=(-0.7, -0.5))
        # plot(x, y, cells, grad_y, title="Gradient Y Component", colorbar_label="Gradient Y", sharp_color_range=(-0.7, -0.5))

        # Extract the normal vectors the upper plate boundary with corresponding midpoints
        normal_vectors_plate, midpoints_plate = _compute_boundary_normals_and_midpoints(domain, boundary_facets)

        # Compute the normal derivative on the plate
        normal_derivatives_plate = grad_x_plate * normal_vectors_plate[:,0] + grad_y_plate * normal_vectors_plate[:,1]
        
        return x, y, cells, potential, grad_x, grad_y, midpoints_plate, normal_derivatives_plate, normal_vectors_plate


def solvensave(mesh: str, data_folder: str = "test") -> None:
    """
    .. admonition:: Description
        
        Full order model that solves the PDE and saves the output in an .h5 file.
    
    :param mesh: path to the mesh file.
    :param data_folder: path to the data folder.
    """
    if mesh.is_file() and mesh.suffix == ".msh":
        x, y, cells, potential, grad_x, grad_y, midpoints_plate, normal_derivatives_plate, normal_vectors_plate = fom(mesh) 
        # Save the results in a .h5 file
        results_folder = Path(data_folder) / "results"
        base_name = os.path.splitext(os.path.basename(mesh))[0]
        filename = results_folder / f"{base_name}.h5"
        with h5py.File(filename, "w") as file:
            file.create_dataset("x", data=x)
            file.create_dataset("y", data=y)
            file.create_dataset("cells", data=cells)
            file.create_dataset("potential", data=potential)
            file.create_dataset("grad_x", data=grad_x)
            file.create_dataset("grad_y", data=grad_y)
            file.create_dataset("normal_derivatives_plate", data=normal_derivatives_plate)
            file.create_dataset("midpoints_plate", data=midpoints_plate)
            file.create_dataset("normal_vectors_plate", data=normal_vectors_plate)