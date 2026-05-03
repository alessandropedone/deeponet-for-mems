import h5py
import numpy as np
import pandas as pd
import os
  
def load(data_folder: str = "test") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    .. admonition:: Description
        
        Load data from all ``.h5`` files in the specified data folder. 
        The function reads the geometrical parameters from a ``.csv`` 
        file and loads the corresponding solution data from ``.h5`` files. 
        It pads the data arrays to ensure uniform lengths across all samples.

    :param data_folder: Path to the data folder.

    :returns:
        - mu (``np.ndarray``) -- Geometrical parameters.
        - x (``np.ndarray``) -- x-coordinates of the domain points.
        - y (``np.ndarray``) -- y-coordinates of the domain points.
    
    :raises FileNotFoundError: If the specified data folder does not exist.
    :raises FileNotFoundError: If the results folder does not exist.
        
    """
    
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"The specified data folder '{data_folder}' does not exist.")
    
    # Empty lists to collect data from all files
    x = []
    y = []
    potential = []
    normal_derivatives_plate = []
    x_plate = []
    y_plate = []

    # Define the results folder path
    results_folder = f"{data_folder}/results"
    
    if not os.path.exists(results_folder):
        raise FileNotFoundError(f"The specified results folder '{results_folder}' does not exist.")

    # Loop through all HDF5 files in the results folder
    for i in range(len(os.listdir(results_folder))):
        path = results_folder + f"/{i+1}.h5"
        with h5py.File(path, "r") as file:
            # Data in the whole domain
            x.append(np.array(file["x"]))
            y.append(np.array(file["y"]))
            potential.append(np.array(file["potential"]))
            # Data on the upper plate
            x_plate.append(np.array(file["midpoints_plate"][:, 0]))
            y_plate.append(np.array(file["midpoints_plate"][:, 1]))
            normal_derivatives_plate.append(np.array(file["normal_derivatives_plate"]))
    
    # Find maximum lengths for padding
    max_vertices = np.max([len(arr) for arr in x])
    max_plate_points = np.max([len(arr) for arr in x_plate])

    # Pad arrays with zeros and nans to ensure uniform length based on maximum lengths
    for i in range(len(x)):
        
        x[i] = np.pad(x[i], (0, max_vertices - len(x[i])), 'constant', constant_values=0)
        y[i] = np.pad(y[i], (0, max_vertices - len(y[i])), 'constant', constant_values=0)
        potential[i] = np.pad(potential[i], (0, max_vertices - len(potential[i])), 'constant', constant_values=np.nan)

        x_plate[i] = np.pad(x_plate[i], (0, max_plate_points - len(x_plate[i])), 'constant', constant_values=0)
        y_plate[i] = np.pad(y_plate[i], (0, max_plate_points - len(y_plate[i])), 'constant', constant_values=0)
        normal_derivatives_plate[i] = np.pad(normal_derivatives_plate[i], (0, max_plate_points - len(normal_derivatives_plate[i])), 'constant', constant_values=np.nan)

    # Stack arrays in each list to create 2D matrixes
    x = np.stack(x)
    y = np.stack(y)
    potential = np.stack(potential)
    x_plate =  np.stack(x_plate)
    y_plate =  np.stack(y_plate)
    normal_derivatives_plate = np.stack(normal_derivatives_plate)

    # Read the geometrical parameters from the CSV file
    data_csv = pd.read_csv(f"{data_folder}/parameters.csv")
    mu = data_csv.iloc[:, 1:]
    mu = np.array(mu)

    return mu, x, y, potential, x_plate, y_plate, normal_derivatives_plate