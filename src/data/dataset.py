from pathlib import Path
from tqdm import tqdm

from .fom import solvensave

def _reset_results(data_folder: str = "test") -> None:
    """
    .. admonition:: Description
        
        Empty the results folder by removing all files inside it.

    :param data_folder: Path to the data folder.
    """
    # Ensure the results folder is empty before running the script
    results_folder = Path(data_folder) / "results"
    if results_folder.exists():
        for file in results_folder.iterdir():
            if file.is_file():
                file.unlink()
    else:
        results_folder.mkdir(parents=True)


def generate_datasets(data_folder: str = "test", empty_results_folder: bool = True) -> None:
    """
    .. admonition:: Description
        
        Generate datasets by processing all mesh files in parallel and saving the results.

    :param data_folder: path to the data folder.
    :param empty_results_folder: Whether to empty the results folder before generating new datasets.

    :raises FileNotFoundError: If the mesh folder does not exist.
    """
    # Set up the environment
    if empty_results_folder:
        _reset_results(data_folder)

    # Get the list of mesh files
    mesh_folder_path = Path(f"{data_folder}/msh")
    if not mesh_folder_path.exists():
        raise FileNotFoundError(f"The specified mesh folder '{mesh_folder_path}' does not exist.")
    meshes = list(mesh_folder_path.iterdir())

    # Process only the meshes that don't have a corresponding results file yet
    results_folder_path = Path(f"{data_folder}/results")
    meshes = [mesh for mesh in meshes if not (results_folder_path / f"{mesh.stem}.h5").exists()]

    for mesh in tqdm(
        meshes,
        total=len(meshes),
        desc="🚀 Generating solution datasets",
        ncols=100,
        bar_format="{desc} |{bar}| {percentage:3.0f}% [{n}/{total}] ⏱️ {elapsed} ETA {remaining}",
        colour='blue'
    ):
        solvensave(mesh, data_folder=data_folder)