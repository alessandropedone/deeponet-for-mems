"""
This module implements the training procedure for a DeepONet surrogate model
that learns the mapping from geometrical parameters and spatial coordinates
to the solution of a physical problem.

Example usage::

    python -m src.surrogate.train --folder "temp/test1" --model_path "models/potential.keras" --target "potential"

There are several optional arguments to customize the behavior:

- ``--folder``: Path to the data folder (default: "test").
- ``--model_path``: Path to save the trained model (default: "models/model.keras").
- ``--r``: Low-rank dimension of the DeepONet (default: :math:`20`).
- ``--seed``: Random seed for data splitting (default: :math:`40`).
- ``--target``: Target quantity to predict, either "potential" or "normal_derivative" (default: "potential").
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import argparse

from .model import DenseNetwork, DeepONet
from .losses import masked_mse, masked_mae
from .loader import load



def train(model_path: str, 
          r: int,
          x: np.ndarray,
          y:  np.ndarray,
          mu: np.ndarray,
          seed: int = 40,
          batch_size: int = 64,
          rescale_output: float = 1.0) -> None:
    """
    .. admonition:: Description

        Train a DeepONet surrogate model to learn the mapping from geometrical parameters 
        and spatial coordinates to the solution of the physical problem.

    :param model_path: Path to save the trained model.
    :param r: Low-rank dimension of the DeepONet.
    :param x: Input coordinates array of shape ``(ns, nh, d)``.
    :param y: Output solution array of shape ``(ns, nh)``.
    :param mu: Geometrical parameters array of shape ``(ns, p)``.
    :param seed: Random seed for data splitting.
    :param batch_size: Batch size for training.
    :param rescale_output: Factor to scale the output during training (default: 1.0, no scaling).
    """
    # Hyperparameters setup ----------------------------------------------------------
    r    = r             # low-rank dimension
    p    = mu.shape[1]   # number of problem parameters = geometrical parameters
    d    = x.shape[2]    # number of spatial dimensions
    ns   = mu.shape[0]   # number of samples = number of meshes
    nh   = x.shape[1]    # max number of dofs = x-coordinates available for each mesh
    seed = seed          # random seed for data splitting
    batch_size = batch_size  # batch size for training
    # --------------------------------------------------------------------------------

    # Print shapes
    print("mu shape:", mu.shape)
    print("x shape:", x.shape)
    print("y shape:", y.shape)

    # Split indices for train+val and test sets first (split along the first dimension)
    idx = np.arange(ns)
    idx_trainval, idx_test = train_test_split(idx, test_size=0.2, random_state=seed)
    # Split train+val indices into train and val sets
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.2, random_state=seed)
    # Use indices to split the arrays along the first dimension
    mu_train, mu_val, mu_test = mu[idx_train], mu[idx_val], mu[idx_test]
    x_train, x_val, x_test = x[idx_train], x[idx_val], x[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    # Print shapes of the splits
    print("mu_train shape:", mu_train.shape)
    print("X_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)

    # Mixed Precision Setup
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)

    branch = DenseNetwork(
        normalization_layer=True,
        input_neurons = p, 
        n_neurons = [128, 64, 32, 128], 
        activation = 'relu', 
        output_neurons = r, 
        output_activation = 'linear', 
        initializer = 'he_normal',
        l1_coeff= 0, 
        l2_coeff = 1e-4, 
        batch_normalization = True, 
        dropout = True, 
        dropout_rate = 0.5, 
        leaky_relu_alpha = None,
        layer_normalization = True,
        positional_encoding_frequencies = 0,
    )
    branch.adapt(mu_train)

    trunk = DenseNetwork(
        normalization_layer=True,
        input_neurons = d, 
        n_neurons = [128, 64, 32, 128], 
        activation = 'relu', 
        output_neurons = r, 
        output_activation = 'linear', 
        initializer = 'he_normal',
        l1_coeff= 0, 
        l2_coeff = 1e-4, 
        batch_normalization = True, 
        dropout = True, 
        dropout_rate = 0.5, 
        leaky_relu_alpha = None,
        layer_normalization = True,
        positional_encoding_frequencies = 10,
    )
    trunk.adapt(x_train)

    model = DeepONet(branch = branch, trunk = trunk, rescale_output=rescale_output)

    model.build(input_shape=[(None, p), (None, d)])
    model.summary()

    # --- Learning rate schedule ---
    def lr_warmup_schedule(epoch, lr):
        warmup_epochs = 5
        base_lr = 1e-3
        start_lr = 1e-6
        if epoch <= warmup_epochs:
            return start_lr + (base_lr - start_lr) * (epoch / warmup_epochs)
        return lr

    warmup_callback = tf.keras.callbacks.LearningRateScheduler(lr_warmup_schedule, verbose=0)

    reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        verbose=1
    )

    model.train_model(
        X = x_train,
        mu = mu_train,
        y = y_train,
        X_val = x_val,
        mu_val = mu_val,
        y_val = y_val,
        learning_rate= 1e-3, 
        epochs = 1000, 
        batch_size = batch_size, 
        loss = masked_mse, 
        validation_freq = 1, 
        verbose = 1, 
        lr_scheduler = [warmup_callback, reduce_callback], 
        metrics = [masked_mae, masked_mse],
        clipnorm = 1, 
        early_stopping_patience = 15,
        log = True,
        optimizer = 'adam')

    print("Evaluating the model on the validation set...")
    model.evaluate([mu_val, x_val], y_val)

    print("Evaluating the model on the test set...")
    model.evaluate([mu_test, x_test], y_test)

    model.save(model_path)

    model.plot_training_history()


def main():
    # Read input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="test", help="Path to the data folder to delete.")
    parser.add_argument("--model_path", type=str, default="models/model.keras", help="Path to save the trained model.")
    parser.add_argument("--r", type=int, default=20, help="Low-rank dimension of the DeepONet.")
    parser.add_argument("--seed", type=int, default=40, help="Random seed for data splitting.")
    parser.add_argument("--target", choices=["potential", "normal_derivative"], default="potential", help="Target quantity to predict.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--rescale_output", type=float, default=1.0, help="Factor to scale the output during training (default: 1.0, no scaling).")
    args = parser.parse_args()
    data_folder = args.folder
    model_path = args.model_path
    r = args.r
    seed = args.seed

    # Import coordinates dataset and solutions
    mu, x, y, potential, x_plate, y_plate, normal_derivatives_plate = load(data_folder=data_folder)

    # Define input and output arrays
    if args.target == "potential":
        x = np.stack((x, y), axis=2)
        y = np.array(potential)
    elif args.target == "normal_derivative":
        x = np.stack((x_plate, y_plate), axis=2)
        y = np.array(normal_derivatives_plate)

    # Train the model
    from .gpu import run_on_device
    run_on_device(train, model_path=model_path, r=r, x=x, y= y, mu=mu, seed=seed, batch_size=args.batch_size, rescale_output=args.rescale_output)

if __name__ == "__main__":
    main()