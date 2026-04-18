import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import BoundaryNorm
import math
import h5py


def _cells_plot(x: np.ndarray,
               y: np.ndarray, 
               cells: np.ndarray, 
               sol: np.ndarray, 
               title: str ="Title", 
               xlabel: str ="x", 
               ylabel: str ="y", 
               colorbar_label: str ="Quantity",
               cmap: str ='RdBu_r',
               sharp_color_range: tuple = None,
               outside_sharpness: int = 50,
               plot_triangulation: bool = True,
               postpone_show: bool = False) -> None:
    """
    .. admonition:: Description
        
        It plots the provided solution over the domain using cell-based data.
        
    :param x: x coordinates of the vertices.
    :param y: y coordinates of the vertices.
    :param cells: Connectivity of the mesh cells.
    :param sol: Quantity defined per cell to be plotted.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param colorbar_label: Label for the color bar.
    :param cmap: Colormap to use for plotting.
    :param sharp_color_range: Optional range to create sharp color transitions.
    :param outside_sharpness: Number of color levels outside the sharp color range.
    :param plot_triangulation: Whether to overlay the mesh triangulation.
    :param postpone_show: Whether to postpone the ``plt.show()`` call.
    """
    
    # Create triangulation object
    triang = tri.Triangulation(x, y, triangles = cells)

    # Plot the solution using tripcolor
    if sharp_color_range is not None:
        # Define custom color normalization with sharp transitions in specified range
        bounds = np.concatenate([
            np.linspace(sol.min(), sharp_color_range[0], outside_sharpness, endpoint=False),
            np.linspace(sharp_color_range[0], sharp_color_range[1], 150),
            np.linspace(sharp_color_range[1], sol.max(), outside_sharpness)
        ])
        norm = BoundaryNorm(boundaries=bounds, ncolors=256, clip=True)
        plt.tripcolor(
            triang,
            facecolors=sol,
            shading='flat',
            cmap=cmap,
            norm=norm,
        )
    else:
        plt.tripcolor(
            triang,
            facecolors=sol,
            shading='flat',
            cmap=cmap
        )
    # plot also the connectivity (mesh)
    if plot_triangulation:
        plt.triplot(triang, color='lightgrey', linewidth=0.5, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label=colorbar_label)
    if not postpone_show:
        plt.show()


def _vertices_plot(x: np.ndarray, 
                  y: np.ndarray,
                  cells: np.ndarray,
                  sol: np.ndarray,
                  title: str ="Title",
                  xlabel: str ="x",
                  ylabel: str ="y",
                  colorbar_label: str ="Quantity",
                  cmap: str ='RdBu_r',
                  sharp_color_range: tuple = None, 
                  outside_sharpness: int = 50,
                  plot_triangulation: bool = True,
                  postpone_show: bool = False) -> None:
    """
    .. admonition:: Description
        
        It plots the specified solution over the domain using vertex-based data.
        
    :param x: x coordinates of the vertices.
    :param y: y coordinates of the vertices.
    :param cells: Connectivity of the mesh cells.
    :param sol: Quantity defined per vertex to be plotted.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param colorbar_label: Label for the color bar.
    :param cmap: Colormap to use for plotting.
    :param sharp_color_range: Optional range to create sharp color transitions.
    :param outside_sharpness: Number of color levels outside the sharp color range.
    :param plot_triangulation: Whether to overlay the mesh triangulation.
    :param postpone_show: Whether to postpone the ``plt.show()`` call.
    """
    
    triang = tri.Triangulation(x, y, triangles = cells)

    if sharp_color_range is not None:
        bounds = np.concatenate([
            np.linspace(sol.min(), sharp_color_range[0], outside_sharpness, endpoint=False),
            np.linspace(sharp_color_range[0], sharp_color_range[1], 150),
            np.linspace(sharp_color_range[1], sol.max(), outside_sharpness)
        ])
        norm = BoundaryNorm(boundaries=bounds, ncolors=256, clip=True)
        plt.tricontourf(
            triang,
            sol,
            levels=256,
            cmap=cmap,
            norm=norm,
        )
    else:
        plt.tricontourf(
            triang,
            sol,
            levels=256,
            cmap=cmap
        )

    # plot also the connectivity (mesh)
    if plot_triangulation:
        plt.triplot(triang, color='lightgrey', linewidth=0.5, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label=colorbar_label)
    if not postpone_show:
        plt.show()


def domain_plot(x: np.ndarray, 
            y: np.ndarray,
            cells: np.ndarray,
            title: str ="Title",
            xlabel: str ="x",
            ylabel: str ="y",
            plot_triangulation: bool = True,
            postpone_show: bool = False) -> None:
    """
    .. admonition:: Description
        
        It plots only the domain and the mesh.
        
    :param x: x coordinates of the vertices.
    :param y: y coordinates of the vertices.
    :param cells: Connectivity of the mesh cells.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param plot_triangulation: Whether to overlay the mesh triangulation.
    :param postpone_show: Whether to postpone the ``plt.show()`` call.
    """
    triang = tri.Triangulation(x, y, triangles = cells)
    triangles = triang.triangles
    neighbors = triang.neighbors

    # plot also the connectivity (mesh)
    if plot_triangulation:
        plt.triplot(triang, color='lightgrey', linewidth=0.5, alpha=0.5)

    boundary_edges = []

    # Each triangle has 3 edges.
    # If a neighbor is -1, that edge is on the boundary.
    for t_idx, tri_nodes in enumerate(triangles):
        for edge_local, neigh in enumerate(neighbors[t_idx]):
            if neigh == -1:  # boundary edge
                i = tri_nodes[edge_local]
                j = tri_nodes[(edge_local + 1) % 3]
                boundary_edges.append((i, j))

    # Remove duplicates while keeping order
    boundary_edges = list(dict.fromkeys(tuple(sorted(e)) for e in boundary_edges))

    # Plot ONLY boundary
    for i, j in boundary_edges:
        plt.plot([x[i], x[j]], [y[i], y[j]], color='black', linewidth=1.0)
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if not postpone_show:
        plt.show()


def plot(x: np.ndarray, 
         y: np.ndarray, 
         cells: np.ndarray, 
         sol: np.ndarray,
         title: str ="Title", 
         xlabel: str ="x", 
         ylabel: str ="y", 
         colorbar_label: str ="Quantity",
         cmap: str ='RdBu_r',
         sharp_color_range: tuple = None,
         outside_sharpness: int = 50,
         plot_triangulation: bool = True,
         postpone_show: bool = False) -> None:
    """
    .. admonition:: Description
        
        It plots the provided solution over the domain.

    :param x: x coordinates of the vertices.
    :param y: y coordinates of the vertices.
    :param cells: Connectivity of the mesh cells.
    :param sol: Quantity defined per vertex or per cell to be plotted.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param colorbar_label: Label for the color bar.
    :param cmap: Colormap to use for plotting.
    :param sharp_color_range: Optional range to create sharp color transitions.
    :param outside_sharpness: Number of color levels outside the sharp color range.
    :param plot_triangulation: Whether to overlay the mesh triangulation.
    :param postpone_show: Whether to postpone the ``plt.show()`` call.
    
    .. note::
        
        If you set the solution to None you can plot only the domain.
    """

    # Check validity of input data
    if len(x) == 0 or len(y) == 0 or len(cells) == 0 or (sol is not None and len(sol) == 0):
        print("No data to plot.")
        return
    
    if len(x) != len(y):
        print("Inconsistent lengths between x and y coordinates.")
        if sol is not None and len(x) != len(sol):
            # check if the solution is given in terms of cells
            if len(cells) % len(sol) != 0:
                print("Inconsistent data lengths between coordinates and solution.")
                return
        return
    
    if sol is None:
        # Plot only the domain
        domain_plot(x, y, cells, title, xlabel, ylabel, plot_triangulation, postpone_show)
    elif len(cells) % len(sol) == 0:
        # Plot using the cells
        _cells_plot(x, y, cells, sol, title, xlabel, ylabel, colorbar_label, cmap, sharp_color_range, outside_sharpness, plot_triangulation, postpone_show)
    else:
        # Plot using the vertices
        _vertices_plot(x, y, cells, sol, title, xlabel, ylabel, colorbar_label, cmap, sharp_color_range, outside_sharpness, plot_triangulation, postpone_show)


def _zoom_around(ax: plt.Axes, x0: float, y0: float, zoom: float) -> None:
    """
    .. admonition:: Description

        It zooms the given axes around the specified center point.
    
    :param ax: The axes object to apply the zoom on.
    :param x0: x coordinate of the zoom center.
    :param y0: y coordinate of the zoom center.
    :param zoom: Zoom factor.
    """
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    width  = (x_max - x_min) / zoom
    height = (y_max - y_min) / zoom

    ax.set_xlim(x0 - width/2,  x0 + width/2)
    ax.set_ylim(y0 - height/2, y0 + height/2)


def plot_domain(file: h5py.File, 
                postpone_show=False, 
                zoom: list[int] = None, 
                center_points: list[tuple] = None) -> None:
    """
    .. admonition:: Description
        
        It plots the domain and the mesh.
        
    :param file: h5py file object containing the solution data.
    :param postpone_show: Whether to postpone the ``plt.show()`` call.
    :param zoom: List of zoom levels for different subplots.
    :param center_points: List of center points :math:`(x_0, y_0)` for each zoom level.
    """
    n = 1 if zoom is None else len(zoom)
    if n > 1:
        rows = max(1, math.floor(math.sqrt(n) * 0.75))
        cols = math.ceil(n / rows)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        axes = [ax]

    if center_points is not None and len(center_points) != n:
        raise ValueError("Length of center_points must match number of zoom levels.")

    for i in range(n):
        plt.sca(axes[i])
        plot(
            x=file["x"][:],
            y=file["y"][:],
            cells=file["cells"][:],
            sol=None,
            title="Domain and mesh (zoom {})".format(zoom[i]) if zoom is not None else "Domain and mesh",
            xlabel="x",
            ylabel="y",
            colorbar_label="",
            cmap='RdBu_r',
            sharp_color_range=None,
            plot_triangulation=True,
            postpone_show=postpone_show
        )
        _zoom_around(
            axes[i],
            x0=center_points[i][0] if center_points is not None else 0,
            y0=center_points[i][1] if center_points is not None else 0,
            zoom=zoom[i] if zoom is not None else 1
        )
        axes[i].set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    if not postpone_show:
        plt.show()
    

def plot_potential(file: h5py.File, 
                   postpone_show=False, 
                   zoom: list[int] = None, 
                   center_points: list[tuple] = None, 
                   pred: bool = False, 
                   error: bool = False, 
                   error_type: str ="se") -> None:
    """
    .. admonition:: Description
        
        It plots the electrostatic potential over the domain.
        
    :param file: h5py file object containing the solution data.
    :param postpone_show: Whether to postpone the ``plt.show()`` call.
    :param zoom: List of zoom levels for different subplots.
    :param center_points: List of center points :math:`(x_0, y_0)` for each zoom level.
    :param pred: Whether to plot the predicted potential.
    :param error: Whether to plot the error in potential prediction.
    :param error_type: Type of error to plot ("se" for squared error, "ae" for absolute error).
    """
    n = 1 if zoom is None else len(zoom)
    if pred and error:
        raise ValueError("Cannot set both pred and error to True.")
    if not error:
        if n > 1:
            rows = max(1, math.floor(math.sqrt(n) * 0.75))
            cols = math.ceil(n / rows)
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        else:
            fig, ax = plt.subplots(figsize=(5, 5))
            axes = [ax]
        if center_points is not None and len(center_points) != n:
            raise ValueError("Length of center_points must match number of zoom levels.")
        for i in range(n):
            plt.sca(axes[i])
            if pred:
                sol = file["potential_pred"][:]
            else:
                sol = file["potential"][:]
            plot(
                x = file["x"][:],
                y = file["y"][:],
                cells = file["cells"][:],
                sol = sol,
                title ="Electrostatic potential (zoom {})".format(zoom[i]) if zoom is not None else "Electrostatic potential", 
                xlabel ="x",
                ylabel ="y",
                colorbar_label ="Potential",
                cmap ='RdBu_r',
                sharp_color_range = None,
                plot_triangulation = True,
                postpone_show = postpone_show
            )
            _zoom_around(
                axes[i],
                x0=center_points[i][0] if center_points is not None else 0,
                y0=center_points[i][1] if center_points is not None else 0,
                zoom=zoom[i] if zoom is not None else 1
            )
            # specifcy its a prediction
            if pred:
                axes[i].set_title("Predicted electrostatic potential (zoom {})".format(zoom[i]) if zoom is not None else "Predicted potential")
            axes[i].set_aspect('equal', adjustable='datalim')
    else:
        if n > 1:
            rows = max(1, math.floor(math.sqrt(n) * 0.75))
            cols = math.ceil(n / rows)
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        else:
            fig, ax = plt.subplots(figsize=(5, 5))
            axes = [ax]
        if center_points is not None and len(center_points) != n:
            raise ValueError("Length of center_points must match number of zoom levels.")
        if error_type not in ["se", "ae"]:
            raise ValueError("error_type must be either 'se' (squared error) or 'ae' (absolute error).")
        if error_type == "se":
            for i in range(n):
                plt.sca(axes[i])
                plot(
                    x = file["x"][:],
                    y = file["y"][:],
                    cells = file["cells"][:],
                    sol = file["se"][:],
                    title ="Squared error (RMSE {:.2e}) (zoom {})".format(np.sqrt(np.mean(file["se"][:])), zoom[i]) if zoom is not None else "Squared error (RMSE {:.2e})".format(np.sqrt(np.mean(file["se"][:]))), 
                    xlabel ="x",
                    ylabel ="y",
                    colorbar_label ="Squared Error",
                    cmap ='RdBu_r',
                    sharp_color_range = None,
                    plot_triangulation = True,
                    postpone_show = postpone_show
                )
                _zoom_around(
                    axes[i],
                    x0=center_points[i][0] if center_points is not None else 0,
                    y0=center_points[i][1] if center_points is not None else 0,
                    zoom=zoom[i] if zoom is not None else 1
                )
                axes[i].set_aspect('equal', adjustable='datalim')
        else:
            for i in range(n):
                plt.sca(axes[i])
                plot(
                    x = file["x"][:],
                    y = file["y"][:],
                    cells = file["cells"][:],
                    sol = file["ae"][:],
                    title ="Absolute error (MAE {:.2e}) (zoom {})".format(np.mean(file["ae"][:]), zoom[i]) if zoom is not None else "Absolute error (MAE {:.2e})".format(np.mean(file["ae"][:])),
                    xlabel ="x",
                    ylabel ="y",
                    colorbar_label ="Absolute Error",
                    cmap ='RdBu_r',
                    sharp_color_range = None,
                    plot_triangulation = True,
                    postpone_show = postpone_show
                )
                _zoom_around(
                    axes[i],
                    x0=center_points[i][0] if center_points is not None else 0,
                    y0=center_points[i][1] if center_points is not None else 0,
                    zoom=zoom[i] if zoom is not None else 1
                )
                axes[i].set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    if not postpone_show:
        plt.show()

    
def plot_grad(file: h5py.File, 
              postpone_show=False, 
              zoom: list[int] = None, 
              center_points: list[tuple] = None, 
              component: str ="x"):
    """
    .. admonition:: Description
        
        It plots the x component of the gradient of the electrostatic potential.
        
    :param file: h5py file object containing the solution data.
    :param postpone_show: Whether to postpone the ``plt.show()`` call.
    :param zoom: List of zoom levels for different subplots.
    :param center_points: List of center points :math:`(x_0, y_0)` for each zoom level.
    :param component: Component of the gradient to plot ("x" or "y").
    """
    n = 1 if zoom is None else len(zoom)
    if n > 1:
        rows = max(1, math.floor(math.sqrt(n) * 0.75))
        cols = math.ceil(n / rows)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        axes = [ax]
    if center_points is not None and len(center_points) != n:
        raise ValueError("Length of center_points must match number of zoom levels.")
    for i in range(n):
        plt.sca(axes[i])
        if component == "x":
            plot(
                x = file["x"][:],
                y = file["y"][:],
                cells = file["cells"][:],
                sol = file["grad_x"][:],
                title ="grad_x (zoom {})".format(zoom[i]) if zoom is not None else "grad_x", 
                xlabel ="x",
                ylabel ="y",
                colorbar_label ="grad_x",
                cmap ='RdBu_r',
                sharp_color_range = None,
                plot_triangulation = True,
                postpone_show = postpone_show
            )
        else:
             plot(
                x = file["x"][:],
                y = file["y"][:],
                cells = file["cells"][:],
                sol = file["grad_y"][:],
                title ="grad_y (zoom {})".format(zoom[i]) if zoom is not None else "grad_y", 
                xlabel ="x",
                ylabel ="y",
                colorbar_label ="grad_y",
                cmap ='RdBu_r',
                sharp_color_range = None,
                plot_triangulation = True,
                postpone_show = postpone_show
            )
        _zoom_around(
            axes[i],
            x0=center_points[i][0] if center_points is not None else 0,
            y0=center_points[i][1] if center_points is not None else 0,
            zoom=zoom[i] if zoom is not None else 1
        )
        axes[i].set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    if not postpone_show:
        plt.show()
   

def plot_normal_derivative(file, postpone_show=False, pred=False, error=False, zoom: list[int] = None, center_points: list[tuple] = None):   
    """
    .. admonition:: Description
        
        It plots the normal derivative of the potential on the upper plate as arrows.
    
    :param file: h5py file object containing the solution data.
    :param postpone_show: Whether to postpone the ``plt.show()`` call.
    :param pred: Whether to plot the predicted normal derivative.
    :param error: Whether to plot the error in normal derivative prediction.
    :param zoom: List of zoom levels for different subplots.
    :param center_points: List of center points :math:`(x_0, y_0)` for each zoom level.
    
    :raises ValueError: If both pred and error are set to True.
    """ 
    n = 1 if zoom is None else len(zoom)
    if n > 1:
        rows = max(1, math.floor(math.sqrt(n) * 0.75))
        cols = math.ceil(n / rows)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        axes = [ax]
    if center_points is not None and len(center_points) != n:
        raise ValueError("Length of center_points must match number of zoom levels.")
    for i in range(n):
        plt.sca(axes[i])
        plot(
            x=file["x"][:],
            y=file["y"][:],
            cells=file["cells"][:],
            sol=None,
            title="Domain and mesh (zoom {})".format(zoom[i]) if zoom is not None else "Domain and mesh",
            xlabel="x",
            ylabel="y",
            colorbar_label="",
            cmap='RdBu_r',
            sharp_color_range=None,
            plot_triangulation=True,
            postpone_show=postpone_show
        )
        _zoom_around(
            axes[i],
            x0=center_points[i][0] if center_points is not None else 0,
            y0=center_points[i][1] if center_points is not None else 0,
            zoom=zoom[i] if zoom is not None else 1
        )
        axes[i].set_aspect('equal', adjustable='datalim')
        if pred and error:
            raise ValueError("Cannot set both pred and error to True.")
        if pred:
            normal_derivative = file["normal_derivative_pred"][:]
        elif error:
            normal_derivative = abs(file["normal_derivative_pred"][:] - file["normal_derivatives_plate"][:]) / np.median(abs(file["normal_derivatives_plate"][:] + 1e-9))
        else:
            normal_derivative = file["normal_derivatives_plate"][:]
        points = file["midpoints_plate"][:]
        normals = file["normal_vectors_plate"][:]
        U = normals[:, 0]
        V = normals[:, 1]
        lengths = normal_derivative
        norm = (lengths - lengths.min()) / (np.ptp(lengths) + 1e-9)  
        cmap = plt.cm.seismic
        for j in range(len(points)):
            color = cmap(norm[j])
            plt.arrow(
                points[j, 0], points[j, 1],
                U[j], V[j],
                head_width=0.05, head_length=0.05,
                fc=color, ec=color
            )
        sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=lengths.min(), vmax=lengths.max())
        )
        sm.set_array([])
        ax = plt.gca()
        plt.colorbar(sm, ax=ax, label="Derivative modulus")
        if pred:
            ax.set_title("Predicted normal derivative (zoom {})".format(zoom[i]) if zoom is not None else "Predicted normal derivative")
        elif error:
            ax.set_title("Normal derivative error rescaled (zoom {})".format(zoom[i]) if zoom is not None else "Normal derivative error rescaled")
        else:
            ax.set_title("Normal derivative (zoom {})".format(zoom[i]) if zoom is not None else "Normal derivative")

    plt.tight_layout()
    if not postpone_show:
        plt.show()


def summary_plot(file: h5py.File) -> None:
    """
    .. admonition:: Description
        
        It creates a summary plot with all relevant plots.
        
    :param file: h5py file object containing the solution data.
    """
    plot_domain(file, postpone_show=True, zoom=[1, 4, 15], center_points=[(0,0), (0,0), (-50,0)])
    plot_potential(file, postpone_show=True, zoom=[1, 4, 15], center_points=[(0,0), (0,0), (-50,0)])
    plot_grad(file, postpone_show=True, zoom=[1, 4, 15], center_points=[(0,0), (0,0), (-50,0)], component="x")
    plot_grad(file, postpone_show=True, zoom=[1, 4, 15], center_points=[(0,0), (0,0), (-50,0)], component="y")
    plot_normal_derivative(file, postpone_show=True, zoom=[4], center_points=[(0,0)])
    plt.show()