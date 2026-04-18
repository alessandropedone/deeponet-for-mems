"""
Evaluate a trained DeepONet surrogate model on a test dataset.

Example usage::
    
    python -m surrogate.evaluate --folder "test/test1" --model_path "models/potential1.keras" --target "potential" 

There are several optional arguments to customize the behavior:

- ``--folder``: Path to the data folder (default: "test").
- ``--model_path``: Path to the trained model file (default: "models/model.keras").
- ``--splitting_seed``: Random seed for data splitting (default: :math:`40`).
- ``--target``: Target quantity to predict, either "potential" or "normal_derivative" (default: "potential").
- ``--prediction_seed``: Random seed for test sample selection (default: :math:`40`).
- ``--using_training_set``: If set, divide the set into training, validation, and test sets as in the training script.

.. note::

    Ensure that the seeds used for splitting and prediction are consistent with those used during training for reproducibility.
"""

def main():
    # Read input arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="test", help="Path to the data folder to delete.")
    parser.add_argument("--model_path", type=str, default="models/model.keras", help="Path to save the trained model.")
    parser.add_argument("--splitting_seed", type=int, default=40, help="Random seed for data splitting.")
    parser.add_argument("--target", choices=["potential", "normal_derivative"], default="potential", help="Target quantity to predict.")
    parser.add_argument("--using_training_set", action="store_true", help="If set, divide the set into training, validation, and test sets as in the training script.")

    args = parser.parse_args()
    data_folder = args.folder
    seed = args.splitting_seed
    target = args.target
    model_path = args.model_path
    training_set_flag = args.using_training_set

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # Import coordinates dataset and solutions
    from .loader import load
    import numpy as np
    mu, x, y, potential, x_plate, y_plate, normal_derivatives_plate = load(data_folder=data_folder)

    if target == "potential":
        x = np.stack((x, y), axis=2)
        y = np.array(potential)
    elif target == "normal_derivative":
        x = np.stack((x_plate, y_plate), axis=2)
        y = np.array(normal_derivatives_plate)

    if training_set_flag:
        # Split the dataset into training, validation, and test sets as in the training script
        from sklearn.model_selection import train_test_split
        ns = mu.shape[0]
        idx = np.arange(ns)
        idx_trainval, idx_test = train_test_split(idx, test_size=0.2, random_state=seed)
        # Split train+val indices into train and val sets
        idx_train, idx_val = train_test_split(idx_trainval, test_size=0.2, random_state=seed)
        # Use indices to split the arrays along the first dimension
        mu_train, mu_val, mu_test = mu[idx_train], mu[idx_val], mu[idx_test]
        x_train, x_val, x_test = x[idx_train], x[idx_val], x[idx_test]
        y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    else:
        mu_test = mu
        x_test = x
        y_test = y

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

    print("\033[38;2;0;175;6m\n\nEvaluating the model on the test set...\033[0m")
    model.evaluate([mu_test, x_test], y_test)

if __name__ == "__main__":
    main()