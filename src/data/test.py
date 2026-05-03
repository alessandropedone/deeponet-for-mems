import h5py

from .geometry import generate_geometries
from .mesh import generate_meshes
from .dataset import generate_datasets
from .plot import summary_plot


def test(names: list[str],
        ranges: list[tuple],
        num_points: list[int],
        geometry_input: str,
        parameters_file_name: str = "parameters.csv",
        data_folder: str = "test", 
        plot_number: int = None,
        empty_old_mesh: bool = True,
        empty_old_results: bool = True,
        workers: int = 1) -> None:
    """
    .. admonition:: Description
        
        Test function to generate geometries, meshes, and datasets, and optionally plot a specific solution.
        It orchestrates the entire process by calling the relevant functions from other modules.
    
    :param names: List of the names, that appear in the geometry file of the quantities.
    :param ranges: List of ranges for each quantity.
    :param num_points: List of number of points to generate within the specified ranges for each quantity.
    :param geometry_input: Path to the input geometry file.
    :param parameters_file_name: name of the parameters file to save the generated parameters.
    :param data_folder: path to the data folder.
    :param plot_number: number of the solution to plot.
    :param empty_old_mesh: whether to empty the old meshes folder before generating new meshes.
    :param empty_old_results: whether to empty the old results folder before generating new results.

    .. note:: 
        
        If ``plot_number`` is ``None``, no plots are generated.

    .. note:: 
        
        The geometry generation is skipped if one of the ``empty_old_mesh`` or ``empty_old_results`` is False.
    """
    if empty_old_mesh and empty_old_results:
        generate_geometries(
            names=names,
            ranges=ranges,
            num_points=num_points,
            geometry_input=geometry_input,
            data_folder=data_folder,
            parameters_file_name=parameters_file_name,
            ignore_data=True,
            max_workers=workers
        )

    generate_meshes(data_folder=data_folder, empty_mesh_folder=empty_old_mesh, max_workers=workers)

    generate_datasets(data_folder=data_folder, empty_results_folder=empty_old_results)

    if plot_number is not None:
        path = data_folder + f"/results/{plot_number}.h5"
        with h5py.File(path, "r") as file:
            summary_plot(file)