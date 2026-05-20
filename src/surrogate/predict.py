"""
This script loads a trained DeepONet surrogate model and uses it to make predictions
on a randomly selected test sample from a specified dataset. It saves the predictions
to a ``.h5`` file and generates plots comparing the surrogate predictions with the full-order
model (FOM) results.

.. note::

    If you sample from the test set, ensure that the splitting seed matches that used during training. 
    If you are using the model we trained just keep the default value of :math:`40`.
"""
import numpy as np
import os
import csv
import argparse

from .loader import load


def _find_id(mu: np.ndarray, 
            data_folder: str = "data") -> int:
    """
    .. admonition:: Description

        Find the index of a test sample in the parameters ``.csv`` file based on its geometrical parameters.

    :param mu: Geometrical parameters of the test sample.
    :param data_folder: Path to the data folder.

    :returns:
        - test_sample_number (``int``) -- The index of the test sample in ``parameters.csv``.

    :raises ValueError: If the test sample mu is not found in ``parameters.csv``.
    """
    parameters_file = os.path.join(data_folder, "parameters.csv")
    with open(parameters_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        mu_list = []
        for row in reader:
            mu_values = [float(value) for value in row]
            # Remove the first element since it's the index
            mu_values = mu_values[1:]
            mu_list.append(mu_values)
    mu_array = np.array(mu_list)
    # Find the index of the test sample in mu_array
    test_sample_number = None
    for i in range(mu_array.shape[0]):
        if np.allclose(mu_array[i], mu):
            test_sample_number = i+1
            break
    if test_sample_number is None:
        raise ValueError("Test sample mu not found in parameters.csv")
    return test_sample_number

def main():
    # Read input arguments    
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="test", help="Path to the data folder to delete.")
    parser.add_argument("--model_path", type=str, default="models/model.keras", help="Path to save the trained model.")
    parser.add_argument("--splitting_seed", type=int, default=40, help="Random seed for data splitting.")
    parser.add_argument("--target", choices=["potential", "normal_derivative"], default="potential", help="Target quantity to predict.")
    parser.add_argument("--prediction_seed", type=int, default=40, help="Random seed for test sample selection.")
    parser.add_argument("--sample_from_test_set", action="store_true", help="If set, sample from the test set; otherwise, sample from the entire dataset.")
    args = parser.parse_args()
    data_folder = args.folder
    seed = args.splitting_seed
    seed2 = args.prediction_seed
    target = args.target
    model_path = args.model_path
    use_test_set = args.sample_from_test_set

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

    # Import coordinates dataset and solutions
    mu, x, y, potential, x_plate, y_plate, normal_derivatives_plate = load(data_folder=data_folder)
    if target == "potential":
        x = np.stack((x, y), axis=2)
        y = np.array(potential)
    elif target == "normal_derivative":
        x = np.stack((x_plate, y_plate), axis=2)
        y = np.array(normal_derivatives_plate)

    import random
    random.seed(seed2)
    if use_test_set:
        # Split the dataset into training, validation, and test sets as in the training script
        from sklearn.model_selection import train_test_split
        ns = mu.shape[0]
        idx = np.arange(ns)
        idx_trainval, idx_test = train_test_split(idx, test_size=0.2, random_state=seed)
        # Identify the index of a random test sample
        test_sample_index = random.choice(idx_test)
    else:
        # Use the entire dataset for choosing the test sample
        # Identify the index of a random sample from the entire dataset
        test_sample_index = random.randint(0, mu.shape[0]-1)

    # Extract the corresponding input and output data
    x_sample = x[test_sample_index:test_sample_index+1]
    y_sample = y[test_sample_index:test_sample_index+1]
    mu_sample = mu[test_sample_index:test_sample_index+1]

    # Load the trained model
    import tensorflow as tf
    from .model import DenseNetwork, FourierFeatures, LogUniformFreqInitializer, EinsumLayer, DeepONet
    from .losses import masked_mse, masked_mae
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={
            "DenseNetwork": DenseNetwork, 
            'FourierFeatures': FourierFeatures, 
            'LogUniformFreqInitializer': LogUniformFreqInitializer, 
            'EinsumLayer': EinsumLayer, 
            'DeepONet': DeepONet,
            'masked_mse': masked_mse,
            'masked_mae': masked_mae,
            })
    print("\033[38;2;0;175;6m\n\nLoaded surrogate model summary.\033[0m")
    model.summary()

    # Make prediction for the selected test sample
    # Print some information
    print("\033[38;2;0;175;6m\n\nTesting the surrogate model on a random test sample.\033[0m")
    y_pred = model([mu_sample,x_sample]).numpy()
    print("Prediction shape:   ", y_pred.shape)
    branch_output = model.call([mu_sample, x_sample], return_branch=True).numpy()
    print("Branch output shape:", branch_output.shape)
    trunk_output = model.call([mu_sample, x_sample], return_trunk=True).numpy()
    print("Trunk output shape: ", trunk_output.shape)

    # Find the index of the test sample in parameters.csv
    idx = _find_id(mu=mu_sample[0], data_folder=data_folder)

    # Read the corresponding .h5 file in data_folder/results and plot the error
    print(f"\033[38;2;0;175;6m\n\nSaving prediction and plotting results for test sample index {idx}.\033[0m")
    print(f"Results file: {os.path.join(data_folder, 'results', f'{idx}.h5')}")
    print(f"\033[38;2;0;175;6m\n\nPlotting prediction and error in the case mu = {mu_sample[0]}.\033[0m")
    print("\n\n")

    if target == "potential":
        import h5py
        from src.data.plot import plot_potential
        import matplotlib.pyplot as plt
        fom_file = os.path.join(data_folder, "results", f"{idx}.h5")
        with h5py.File(fom_file, 'a') as file:
            # use x[0] to dectect and remove nan values in y_pred
            nan_mask = ~np.isnan(y_sample[0])
            if "potential_pred" in file:
                del file["potential_pred"]
            file["potential_pred"] = y_pred[0][nan_mask]
            if "se" in file:
                del file["se"]
            file["se"] = (y_pred[0][nan_mask] - file["potential"][:])**2
            if "ae" in file:
                del file["ae"]
            file["ae"] = np.abs(y_pred[0][nan_mask] - file["potential"][:])
            plot_potential(file, postpone_show=True, zoom=[1, 15, 15], center_points=[(0,0), (-50,0), (50,0)])
            plot_potential(file, postpone_show=True, zoom=[1, 15, 15], center_points=[(0,0), (-50,0), (50,0)], error = True, error_type='ae')
            plot_potential(file, postpone_show=True, zoom=[1, 15, 15], center_points=[(0,0), (-50,0), (50,0)], error = True, error_type='se')
            plot_potential(file, postpone_show=True, zoom=[1, 15, 15], center_points=[(0,0), (-50,0), (50,0)], pred = True)
            plt.show()

    elif target == "normal_derivative":
        import h5py
        from src.data.plot import plot_normal_derivative
        import matplotlib.pyplot as plt
        fom_file = os.path.join(data_folder, "results", f"{idx}.h5")
        with h5py.File(fom_file, 'a') as file:
            # use x[0] to dectect and remove nan values in y_pred
            nan_mask = ~np.isnan(y_sample[0])
            if "normal_derivative_pred" in file:
                del file["normal_derivative_pred"]
            file["normal_derivative_pred"] = y_pred[0][nan_mask]
            if "normal_se" in file:
                del file["normal_se"]
            file["normal_se"] = (y_pred[0][nan_mask] - file["normal_derivatives_plate"][:])**2
            if "normal_ae" in file:
                del file["normal_ae"]
            file["normal_ae"] = np.abs(y_pred[0][nan_mask] - file["normal_derivatives_plate"][:])
            plot_normal_derivative(file, postpone_show=True, zoom=[4, 15, 15], center_points=[(0,0), (-50,0), (50,0)])
            plot_normal_derivative(file, postpone_show=True, zoom=[4, 15, 15], center_points=[(0,0), (-50,0), (50,0)], error = True)
            plot_normal_derivative(file, postpone_show=True, zoom=[4, 15, 15], center_points=[(0,0), (-50,0), (50,0)], pred = True)
            plt.show()


if __name__ == "__main__":
    main()