"""
Script to generate meshes, run simulations, and save results based on geometry parameters.

Example usage::

    python -m data.generate --folder "test/test1" --data_file "test/test1.csv" --geometry_input "geometries/cantilever1.geo" --workers 2

There are several optional arguments to customize the behavior:

- ``--folder``: Path to the data folder (default: "test").
- ``--keep_old_mesh``: If set, the old meshes folder will not be emptied before generating new meshes.
- ``--keep_old_results``: If set, the old results folder will not be emptied before generating new results.
- ``--data_file``: Path to the data file containing geometry parameter specifications (default: "test.csv").
- ``--parameters_file_name``: Name of the parameters file to save the generated parameters (default: "parameters.csv").
- ``--geometry_input``: Path to the input geometry file (default: "geometry.geo").
- ``--plot_number``: Number of the solution to plot (default: 1).
- ``--workers``: Number of worker processes to use for parallel mesh generation (default: 1).
"""

import argparse

def main():
    parser = argparse.ArgumentParser()


    parser.add_argument("--folder", type=str, default="test", help="Path to the data folder to delete.")
    parser.add_argument("--keep_old_mesh", action="store_true", help="Whether to empty the old meshes folder before generating new meshes.")
    parser.add_argument("--keep_old_results", action="store_true", help="Whether to empty the old results folder before generating new results.")
    parser.add_argument("--data_file", type=str, default="test.csv", help="Path to the data file containing geometry parameter specifications.")
    parser.add_argument("--parameters_file_name", type=str, default="parameters.csv", help="Name of the parameters file to save the generated parameters.")
    parser.add_argument("--geometry_input", type=str, default="geometry.geo", help="Path to the input geometry file.")
    parser.add_argument("--plot_number", type=int, default=1, help="Number of the solution to plot.")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes to use for parallel mesh generation.")

    data_folder = parser.parse_args().folder
    empty_old_mesh = not parser.parse_args().keep_old_mesh
    empty_old_results = not parser.parse_args().keep_old_results
    data_file = parser.parse_args().data_file
    parameters_file_name = parser.parse_args().parameters_file_name
    geometry_input = parser.parse_args().geometry_input
    plot_number = parser.parse_args().plot_number
    workers = parser.parse_args().workers

    from .geometry import read_data_file
    names, ranges, num_points = read_data_file(data_file)

    from .test import test
    test(
        names=names,
        ranges=ranges,
        num_points=num_points,
        geometry_input=geometry_input,
        parameters_file_name=parameters_file_name,
        data_folder=data_folder,
        plot_number=plot_number,
        empty_old_mesh=empty_old_mesh,
        empty_old_results=empty_old_results,
        workers=workers,
    )


if __name__ == "__main__":
    main()


