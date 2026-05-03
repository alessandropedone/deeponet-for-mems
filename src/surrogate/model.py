import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import datetime
from keras.saving import register_keras_serializable
    
@register_keras_serializable()
class FourierFeatures(tf.keras.layers.Layer):
    """
    .. admonition:: Description

        A Keras Layer that applies Fourier Feature Mapping to the input.

        Fourier Feature Mapping is a technique used to embed inputs into a 
        higher-dimensional space using sinusoidal functions. This layer can 
        use either fixed or learnable frequencies.

        Given an input tensor `x` of shape ``(batch_size, ..., input_dim)``, it computes the Fourier features as follows:

        1. For each frequency :math:`f_i` in the set of frequencies, compute :math:`\sin(f_i x_j)` and :math:`\cos(f_i x_j)` 
        for each dimension :math:`x_j` of the input.
        2. Concatenate all the sine and cosine features along the last dimension.

        Thus, the output tensor will have shape ``(batch_size, ..., input_dim + 2 * input_dim * num_frequencies)``.

    :param num_frequencies: Number of frequency bands used for encoding.
    :type num_frequencies: ``int``
    :param learnable: Whether the frequencies are trainable.
    :type learnable: ``bool``
    :param initializer: Initializer used for learnable frequencies.
    :type initializer: ``str`` or ``tf.keras.initializers.Initializer``

    :ivar freqs: Tensor containing the Fourier frequencies used for encoding.
    :vartype freqs: ``tf.Variable`` or ``tf.Tensor``
    :ivar num_frequencies: Number of frequency bands used for encoding.
    :vartype num_frequencies: ``int``
    :ivar learnable: Whether the frequencies are trainable.
    :vartype learnable: ``bool``
    :ivar initializer: Initializer used for learnable frequencies.
    :vartype initializer: ``str`` or ``tf.keras.initializers.Initializer``

    :raises ValueError: If ``num_frequencies`` is not a positive integer.
    """
    def __init__(self, num_frequencies: int, learnable: bool = True, initializer = 'glorot_uniform', **kwargs):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.learnable = learnable
        self.initializer = initializer

    def build(self, input_shape: tf.TensorShape):
        """
        Builds the layer by initializing frequencies.
        
        :param input_shape: Shape of the input tensor.
        """
        shape = (1, self.num_frequencies)
        if self.learnable:
            self.freqs = self.add_weight(name="freqs", shape=shape,
                                         initializer=self.initializer,
                                         trainable=True)
        else:
            self.freqs = tf.constant(2.0 ** tf.range(1, self.num_frequencies + 1, dtype=tf.float32)[tf.newaxis, :])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Applies Fourier Feature Mapping to the input tensor.

        :param x: Input tensor of shape ``(batch_size, ..., input_dim)``.
        :returns:
            - encoded (``tf.Tensor``) -- The Fourier feature encoded tensor.
        """
        # Start encoding list
        encoded = []
        # Apply sinusoidal or learnable frequency encoding
        for i in range(self.num_frequencies):
            freq = self.freqs[0, i]
            for j in range(x.shape[-1]): # x shape: (batch, time, feature)
                current_x = x[..., j:j+1]  # shape: (batch, time, 1)
                encoded.append(tf.sin(freq * current_x))
                encoded.append(tf.cos(freq * current_x))
        # Concatenate along the feature dimension
        encoded_features = tf.concat(encoded, axis=-1)
        # Combine the original x with the new encodings
        return tf.concat([x, encoded_features], axis=-1)
    
    def get_config(self):
        """
        Returns the configuration of the layer for serialization.
        
        :returns:
            - config (``dict``) -- Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "num_frequencies": self.num_frequencies,
            "learnable": self.learnable,
            # For initializer, save its config dict if possible
            "initializer": tf.keras.initializers.serialize(self.initializer),
        })
        return config
    

@register_keras_serializable()
class LogUniformFreqInitializer(tf.keras.initializers.Initializer):
    """
    .. admonition:: Description

        Initializer that generates frequencies sampled from a log-uniform distribution.

    :param min_exp: Minimum exponent for the log-uniform distribution.
    :type min_exp: ``float``
    :param max_exp: Maximum exponent for the log-uniform distribution.
    :type max_exp: ``float``
    :ivar min_exp: Minimum exponent for the log-uniform distribution.
    :vartype min_exp: ``float``
    :ivar max_exp: Maximum exponent for the log-uniform distribution.
    :vartype max_exp: ``float``
    """
    def __init__(self, min_exp=0.0, max_exp=8.0):
        self.min_exp = min_exp
        self.max_exp = max_exp
        
    def __call__(self, shape, dtype=None):
        # Sample uniformly from [min_exp, max_exp]
        exponents = tf.random.uniform(shape, self.min_exp, self.max_exp, dtype=dtype)
        return tf.math.pow(2.0, exponents)

    def get_config(self):
        """
        Returns the configuration of the initializer for serialization.

        :returns:
            - config (``dict``) -- Configuration dictionary.
        """
        return {'min_exp': self.min_exp, 'max_exp': self.max_exp}


@register_keras_serializable()
class DenseNetwork(tf.keras.Model):
    """
    .. admonition:: Description
        
        A class to build, train, and manage a neural network model using TensorFlow and Keras.

    :param normalization_layer: Whether to include a normalization layer at the input.
    :type normalization_layer: ``bool``
    :param input_neurons: Number of input neurons.
    :type input_neurons: ``int``
    :param n_neurons: List specifying the number of neurons in each hidden layer.
    :type n_neurons: ``list``
    :param activation: Activation function for hidden layers.
    :type activation: ``str``
    :param output_neurons: Number of output neurons.
    :type output_neurons: ``int``
    :param output_activation: Activation function for the output layer.
    :type output_activation: ``str``
    :param initializer: Initializer for the weights of the layers.
    :type initializer: ``str``
    :param l1_coeff: L1 regularization coefficient.
    :type l1_coeff: ``float``
    :param l2_coeff: L2 regularization coefficient.
    :type l2_coeff: ``float``
    :param batch_normalization: Whether to include batch normalization layers after hidden layers.
    :type batch_normalization: ``bool``
    :param dropout: Whether to include dropout layers after hidden layers.
    :type dropout: ``bool``
    :param dropout_rate: Dropout rate for dropout layers.
    :type dropout_rate: ``float``
    :param leaky_relu_alpha: Alpha parameter for LeakyReLU activation. If None, standard activation is used.
    :type leaky_relu_alpha: ``float``
    :param layer_normalization: Whether to include layer normalization after hidden layers.
    :type layer_normalization: ``bool``
    :param positional_encoding_frequencies: Number of frequencies for Fourier feature positional encoding.
    :type positional_encoding_frequencies: ``int``
    :ivar normalization_layer: Whether to include a normalization layer at the input.
    :vartype normalization_layer: ``bool``
    :ivar norm_layer: Keras Normalization layer.
    :vartype norm_layer: ``tf.keras.layers.Normalization``
    :ivar input_neurons: Number of input neurons.
    :vartype input_neurons: ``int``
    :ivar n_neurons: List specifying the number of neurons in each hidden layer.
    :vartype n_neurons: ``list``
    :ivar activation: Activation function for hidden layers.
    :vartype activation: ``str``
    :ivar output_neurons: Number of output neurons.
    :vartype output_neurons: ``int``
    :ivar output_activation: Activation function for the output layer.
    :vartype output_activation: ``str``
    :ivar initializer: Initializer for the weights of the layers.
    :vartype initializer: ``str``
    :ivar l1_coeff: L1 regularization coefficient.
    :vartype l1_coeff: ``float``
    :ivar l2_coeff: L2 regularization coefficient.
    :vartype l2_coeff: ``float``
    :ivar batch_normalization: Whether
    :vartype batch_normalization: ``bool``
    :ivar dropout: Whether to include dropout layers after hidden layers.
    :vartype dropout: ``bool``
    :ivar dropout_rate: Dropout rate for dropout layers.
    :vartype dropout_rate: ``float``
    :ivar leaky_relu_alpha: Alpha parameter for LeakyReLU activation. If None, standard activation is used.
    :vartype leaky_relu_alpha: ``float``
    :ivar layer_normalization: Whether to include layer normalization after hidden layers.
    :vartype layer_normalization: ``bool``
    :ivar positional_encoding_frequencies: Number of frequencies for Fourier feature positional encoding.
    :vartype positional_encoding_frequencies: ``int``
    :ivar all_layers: List of all layers in the model.
    :vartype all_layers: ``list``
    :ivar history: Training history after model training.
    :vartype history: ``tf.keras.callbacks.History``
    :raises ValueError: If any of the input parameters are invalid.
    """

    ##
    def __init__(self,
                normalization_layer: bool = True,
                input_neurons: int = 1, 
                n_neurons: list = None, 
                activation: str = 'tanh',
                output_neurons: int = 1,
                output_activation: str = 'linear',
                initializer: str = 'glorot_uniform',
                l1_coeff: float = 0,
                l2_coeff: float = 0,
                batch_normalization: bool = False,
                dropout: bool = False,
                dropout_rate: float = 0.3,
                leaky_relu_alpha: float = None,
                layer_normalization: bool = False,
                positional_encoding_frequencies: int = 0,
                **kwargs):

        super().__init__(**kwargs)

        self.normalization_layer = normalization_layer
        self.norm_layer = tf.keras.layers.Normalization(axis=-1)
        self.input_neurons = input_neurons
        self.n_neurons = n_neurons or [64] * 8  # safe default
        self.activation = activation
        self.output_neurons = output_neurons
        self.output_activation = output_activation
        self.initializer = initializer
        self.l1_coeff = l1_coeff
        self.l2_coeff = l2_coeff
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.leaky_relu_alpha = leaky_relu_alpha
        self.layer_normalization = layer_normalization
        self.positional_encoding_frequencies = positional_encoding_frequencies
        self.all_layers = list()

        # Initialize history to None
        self.history = None


    def build(self, input_shape):
        """
        Builds the neural network architecture based on the provided configuration.

        :param input_shape: Shape of the input tensor.
        """
        l1_l2 = tf.keras.regularizers.l1_l2
        Dense = tf.keras.layers.Dense
        BatchNormalization = tf.keras.layers.BatchNormalization
        Dropout = tf.keras.layers.Dropout
        LeakyReLU = tf.keras.layers.LeakyReLU

        self.all_layers = []

        # Normalization
        if self.normalization_layer:
            self.all_layers.append(self.norm_layer)

        # Positional Encoding
        if self.positional_encoding_frequencies and self.positional_encoding_frequencies > 0:
            self.all_layers.append(FourierFeatures(
                num_frequencies=self.positional_encoding_frequencies, 
                learnable=True, 
                initializer=LogUniformFreqInitializer(min_exp=0.0, max_exp=8.0)
            ))

        # First hidden layer
        if self.leaky_relu_alpha is not None:
            self.all_layers.append(Dense(
                self.n_neurons[0], 
                kernel_initializer=self.initializer,
                kernel_regularizer=l1_l2(l1=self.l1_coeff, l2=self.l2_coeff)
            ))
            self.all_layers.append(LeakyReLU(alpha=self.leaky_relu_alpha))
        else:
            self.all_layers.append(Dense(
                self.n_neurons[0], 
                activation=self.activation,
                kernel_initializer=self.initializer,
                kernel_regularizer=l1_l2(l1=self.l1_coeff, l2=self.l2_coeff)
            ))
        
        if self.batch_normalization:
            self.all_layers.append(BatchNormalization())
        if self.dropout:
            self.all_layers.append(Dropout(self.dropout_rate))

        # Hidden layers
        for neurons in self.n_neurons[1:]:
            if self.leaky_relu_alpha is not None:
                self.all_layers.append(Dense(
                    neurons, 
                    kernel_initializer=self.initializer,
                    kernel_regularizer=l1_l2(l1=self.l1_coeff, l2=self.l2_coeff)
                ))
                self.all_layers.append(LeakyReLU(alpha=self.leaky_relu_alpha))
            else:
                self.all_layers.append(Dense(
                    neurons, 
                    activation=self.activation,
                    kernel_initializer=self.initializer,
                    kernel_regularizer=l1_l2(l1=self.l1_coeff, l2=self.l2_coeff)
                ))

            if self.batch_normalization:
                self.all_layers.append(BatchNormalization())
            if self.dropout:
                self.all_layers.append(Dropout(self.dropout_rate))
            if self.layer_normalization:
                self.all_layers.append(tf.keras.layers.LayerNormalization())
        
        # Output layer
        self.all_layers.append(Dense(
            self.output_neurons, 
            activation=self.output_activation,
            kernel_regularizer=l1_l2(l1=self.l1_coeff, l2=self.l2_coeff)
        ))

        dummy_input = tf.keras.Input(shape=(self.input_neurons,))
        self.call(dummy_input)

        super(DenseNetwork, self).build(input_shape)

    def call(self, x):
        """
        Forward pass through the neural network.

        :param x: Input tensor.
        :returns:
            - x (``tf.Tensor``) -- Output tensor after passing through the network.
        """
        for layer in self.all_layers:
            x = layer(x)
        return x
    
    def adapt(self, X: np.ndarray):
        """
        Adapts the normalization layer to the data.

        :param X: Input data for adaptation.
        """
        if self.normalization_layer:
            if X.ndim == 3:
                self.norm_layer.adapt(X.reshape(-1, X.shape[-1]))
            else:
                self.norm_layer.adapt(X)
    
    def get_config(self):
        """
        Returns the configuration of the model for serialization.

        :returns:
            - config (``dict``) -- Configuration dictionary.
        """
        # Return the config necessary to reconstruct this model
        base_config = super(DenseNetwork, self).get_config()
        return {
            **base_config,
            "normalization_layer": self.normalization_layer,
            "input_neurons": self.input_neurons,
            "n_neurons": self.n_neurons,
            "activation": self.activation,
            "output_neurons": self.output_neurons,
            "output_activation": self.output_activation,
            "initializer": self.initializer,
            "l1_coeff": self.l1_coeff,
            "l2_coeff": self.l2_coeff,
            "batch_normalization": self.batch_normalization,
            "dropout": self.dropout,
            "dropout_rate": self.dropout_rate,
            "leaky_relu_alpha": self.leaky_relu_alpha,
            "layer_normalization": self.layer_normalization,
            "positional_encoding_frequencies": self.positional_encoding_frequencies,
        }
    
    @classmethod
    def from_config(cls, config):
        """
        Creates a model instance from the configuration dictionary.

        :param config: Configuration dictionary.
        :returns:
            - model (``DenseNetwork``) -- Model instance created from the configuration.
        """
        return cls(**config)
        
    def train_model(self, 
                    X: np.ndarray, 
                    y: np.ndarray, 
                    X_val: np.ndarray, 
                    y_val: np.ndarray, 
                    learning_rate: float = 1e-3, 
                    epochs: int = 10000, 
                    batch_size: int = 15000, 
                    loss: str = 'mean_squared_error', 
                    validation_freq: int = 1, 
                    verbose: int = 0,
                    lr_scheduler = None,
                    metrics: list = ['mse'],
                    clipnorm: float = None,
                    early_stopping_patience: int = None,
                    log: bool = False,
                    optimizer: str = 'adam'
                    ) -> None:
        """
        Trains the model on the provided dataset.

        :param X: The input data for training.
        :param y: The target data for training.
        :param X_val: The input data for validation.
        :param y_val: The target data for validation.
        :param learning_rate: The learning rate for the optimizer.
        :param epochs: The number of epochs for training.
        :param batch_size: The size of the batches for training.
        :param loss: The loss function to be used during training.
        :param validation_freq: The frequency of validation during training.
        :param lr_scheduler: A list of learning rate scheduler callbacks.
        :param metrics: List of metrics to be evaluated by the model during training and testing.
        :param clipnorm: Gradient clipping norm value. If None, no clipping is applied.
        :param early_stopping_patience: Number of epochs with no improvement after which training will be stopped. If None, early stopping is not used.
        :param log: Whether to log training progress for TensorBoard.
        :param optimizer: The optimizer to be used for training. Options are 'adam', 'sgd', 'rmsprop'.

        .. note::

            Use ``tensorboard --logdir logs`` to visualize logs (if log is set to True).
        """
        if X.size == 0 or y.size == 0 or X_val.size == 0 or y_val.size == 0:
            raise ValueError("Input arrays must not be empty")
        if loss == 'huber_loss':
            loss = tf.keras.losses.Huber(delta=1.0)

        if optimizer not in ['adam', 'sgd', 'rmsprop']:
            raise ValueError("Unsupported optimizer. Supported optimizers are: 'adam', 'sgd', 'rmsprop'.")
        if optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD
            if clipnorm is not None:
                self.compile(loss=loss, metrics=metrics,optimizer=optimizer(learning_rate=learning_rate, momentum=0.9, nesterov=True, clipnorm=clipnorm))
            else:
                self.compile(loss=loss, metrics=metrics,optimizer=optimizer(learning_rate=learning_rate, momentum=0.9, nesterov=True))
        elif optimizer == 'rmsprop':
            self.compile(loss=loss, metrics=metrics,optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.9, epsilon=1e-07, centered=False))
        elif optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam
            if clipnorm is not None:
                self.compile(loss=loss, metrics=metrics,optimizer=optimizer(learning_rate=learning_rate, clipnorm=clipnorm))
            else:
                self.compile(loss=loss, metrics=metrics,optimizer=optimizer(learning_rate=learning_rate))
        
        callbacks = []
        if lr_scheduler is not None:
            for callback in lr_scheduler:
                callbacks.append(callback)

        # Set up TensorBoard callback with profiling
        if log:
            log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_steps_per_second=True)
            callbacks.append(tensorboard_callback)
        # TensorBoard command: tensorboard --logdir logs

        # Early stopping callback
        if early_stopping_patience is not None:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=early_stopping_patience, 
                restore_best_weights=True
            )
            callbacks.append(early_stopping)

        self.history = self.fit(
            X, y, epochs=epochs, batch_size=batch_size, verbose=verbose,
            validation_data=(X_val, y_val), validation_freq=validation_freq,
            callbacks=callbacks
        )

    ##
    def plot_training_history(self) -> None:
        """
        Plots the training and validation loss over epochs.
        This method should be called after training the model using ``train_model``.

        :raises ValueError: If the model has no training history.
        """
        if self.history is None:
            raise ValueError("The model has no training history. Train the model using 'train_model' method first.")

        plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
        tot_train = len(self.history.history['loss'])
        tot_valid = len(self.history.history['val_loss']) 
        valid_freq = int(tot_train / tot_valid)
        plt.plot(np.arange(tot_train), self.history.history['loss'], 'b-', label='Training loss', linewidth=2)
        plt.plot(valid_freq * np.arange(tot_valid), self.history.history['val_loss'], 'r--', label='Validation loss', linewidth=2)
        plt.yscale('log')
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.title('Training and Validation Loss', fontsize=16)
        plt.grid(True)
        plt.show()

@register_keras_serializable()
class EinsumLayer(tf.keras.layers.Layer):
    """
    .. admonition:: Description

        A Keras Layer that performs Einstein summation on input tensors using a specified einsum syntax.

    :param ein_syntax: The Einstein summation syntax string.
    :type ein_syntax: ``str``
    :ivar ein_syntax: The Einstein summation syntax string.
    :vartype ein_syntax: ``str``
    """
    def __init__(self, ein_syntax: str, **kwargs):
        super().__init__(**kwargs)
        self.ein_syntax = ein_syntax

    def call(self, inputs):
        """
        Performs Einstein summation on the input tensors.

        :param inputs: A list or tuple of input tensors to be summed.
        :returns:
            - result (``tf.Tensor``) -- The result of the Einstein summation.
        """
        coeffs, basis = inputs
        return tf.einsum(self.ein_syntax, coeffs, basis)
    
    def get_config(self):
        """
        Returns the configuration of the layer for serialization.

        :returns:
            - config (``dict``) -- Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "ein_syntax": self.ein_syntax,
        })
        return config
    
@register_keras_serializable()
class DeepONet(tf.keras.Model):
    """
    .. admonition:: Description

        A Deep Operator Network (DeepONet) model that combines a branch network and a trunk network
        to approximate operators mapping functions to functions.

    :param branch: The branch network to generate expansion coefficients.
    :type branch: ``DenseNetwork``
    :param trunk: The trunk network to generate the basis functions.
    :type trunk: ``DenseNetwork``
    :ivar branch: The branch network to generate expansion coefficients.
    :vartype branch: ``DenseNetwork``
    :ivar trunk: The trunk network to generate the basis functions.
    :vartype trunk: ``DenseNetwork``
    :ivar history: Training history after model training.
    :vartype history: ``tf.keras.callbacks.History``
    :raises AssertionError: If the output dimensions of the branch and trunk networks do not match
    """

    def __init__(
            self, 
            branch : DenseNetwork, 
            trunk : DenseNetwork,
            **kwargs
        ):
        super(DeepONet, self).__init__(**kwargs)
        assert branch.output_neurons == trunk.output_neurons
        self.branch = branch
        self.trunk = trunk

        # Initialize history to None
        self.history = None

    def call(self, inputs, return_branch=False, return_trunk=False):
        """ 
        Forward pass through the DeepONet model.

        :param inputs: A tuple containing (mu, x) where:
            - mu: batch of parameters (dimension: B x p)
            - x: batch of spatial points (tensor whose last dimension is d)
        :param return_branch: If True, returns the branch output instead of the full DeepONet output.
        :param return_trunk: If True, returns the trunk output instead of the full DeepONet output.
        :returns:
            - output (``tf.Tensor``) -- The output of the DeepONet model or the branch/trunk outputs based on flags.
        """
        mu, x = inputs

        # Compute branch output
        # Input mu shape: batch x p
        # Output shape: batch x low rank dimension
        coeffs = self.branch(mu)
        
        # Compute trunk output
        # Input x shape: time x d  OR  batch x time x d
        # Output shape: time x low rank dimension  OR  batch x time x low rank dimension
        if len(x.shape) == 3:  # batch x time x d
            # Apply trunk to each time step
            original_shape = tf.shape(x)
            x_reshaped = tf.reshape(x, [-1, x.shape[-1]])
            basis_flat = self.trunk(x_reshaped)
            basis = tf.reshape(basis_flat, [original_shape[0], original_shape[1], -1])
        else:  # time x d
            basis = self.trunk(x)

        if return_branch & return_trunk:
            return coeffs, basis
        if return_branch:
            return coeffs
        if return_trunk:
            return basis
        
        ein_syntax = 'bj,bij->bi' if (len(basis.shape) == 3) else 'bj,ij->bi'
        output = EinsumLayer(ein_syntax)([coeffs, basis])
        return output

    def build(self, input_shape):
        """
        Builds the DeepONet model by initializing the branch and trunk networks.

        :param input_shape: Shape of the input tensor.
        """
        dummy_mu = tf.keras.Input(shape=(self.branch.input_neurons,))
        dummy_x = tf.keras.Input(shape=(self.trunk.input_neurons,))
        self.call((dummy_mu, dummy_x))
        super(DeepONet, self).build(input_shape)
    
    def get_config(self):
        """
        Returns the configuration of the model for serialization.

        :returns:
            - config (``dict``) -- Configuration dictionary.
        """
        # Return the config necessary to reconstruct this model
        base_config = super(DeepONet, self).get_config()
        return {
            **base_config,
            "branch": tf.keras.utils.serialize_keras_object(self.branch),
            "trunk": tf.keras.utils.serialize_keras_object(self.trunk)
        }

    @classmethod
    def from_config(cls, config):
        """
        Creates a model instance from the configuration dictionary.

        :param config: Configuration dictionary.
        :returns:
            - model (``DeepONet``) -- Model instance created from the configuration.
        """
        branch_config = config.pop("branch")
        trunk_config = config.pop("trunk")
        config["branch"] = tf.keras.utils.deserialize_keras_object(branch_config)
        config["trunk"] = tf.keras.utils.deserialize_keras_object(trunk_config)
        return cls(**config)
    

    def train_model(self, 
                    X: np.ndarray,
                    mu: np.ndarray,
                    y: np.ndarray, 
                    X_val: np.ndarray, 
                    mu_val: np.ndarray,
                    y_val: np.ndarray, 
                    learning_rate: float = 1e-3, 
                    epochs: int = 10000, 
                    batch_size: int = 15000, 
                    loss: str = 'mean_squared_error', 
                    validation_freq: int = 1, 
                    verbose: int = 0,
                    lr_scheduler = None,
                    metrics: list = ['mse'],
                    clipnorm: float = None,
                    early_stopping_patience: int = None,
                    log: bool = False,
                    optimizer: str = 'adam'
                    ) -> None:
        """
        Trains the model on the provided dataset.

        :param X: The input data for training.
        :param mu: The parameter data for training.
        :param y: The target data for training.
        :param X_val: The input data for validation.
        :param mu_val: The parameter data for validation.
        :param y_val: The target data for validation.
        :param learning_rate: The learning rate for the optimizer.
        :param epochs: The number of epochs for training.
        :param batch_size: The size of the batches for training.
        :param loss: The loss function to be used during training.
        :param validation_freq: The frequency of validation during training.
        :param lr_scheduler: A list of learning rate scheduler callbacks.
        :param metrics: List of metrics to be evaluated by the model during training and testing.
        :param clipnorm: Gradient clipping norm value. If None, no clipping is applied.
        :param early_stopping_patience: Number of epochs with no improvement after which training will be stopped. If None, early stopping is not used.
        :param log: Whether to log training progress for TensorBoard.
        :param optimizer: The optimizer to be used for training. Options are 'adam', 'sgd', 'rmsprop'.

        .. note::
            
            Use ``tensorboard --logdir logs`` to visualize logs (if log is set to True).
        """
        if X.size == 0 or y.size == 0 or X_val.size == 0 or y_val.size == 0 or mu.size == 0 or mu_val.size == 0:
            raise ValueError("Input arrays must not be empty")
        if loss == 'huber_loss':
            loss = tf.keras.losses.Huber(delta=1.0)

        if optimizer not in ['adam', 'sgd', 'rmsprop']:
            raise ValueError("Unsupported optimizer. Supported optimizers are: 'adam', 'sgd', 'rmsprop'.")
        if optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD
            if clipnorm is not None:
                self.compile(loss=loss, metrics=metrics,optimizer=optimizer(learning_rate=learning_rate, momentum=0.9, nesterov=True, clipnorm=clipnorm))
            else:
                self.compile(loss=loss, metrics=metrics,optimizer=optimizer(learning_rate=learning_rate, momentum=0.9, nesterov=True))
        elif optimizer == 'rmsprop':
            self.compile(loss=loss, metrics=metrics,optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.9, epsilon=1e-07, centered=False))
        elif optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam
            if clipnorm is not None:
                self.compile(loss=loss, metrics=metrics,optimizer=optimizer(learning_rate=learning_rate, clipnorm=clipnorm))
            else:
                self.compile(loss=loss, metrics=metrics,optimizer=optimizer(learning_rate=learning_rate))
        
        callbacks = []
        if lr_scheduler is not None:
            for callback in lr_scheduler:
                callbacks.append(callback)

        # Set up TensorBoard callback with profiling
        if log:
            log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_steps_per_second=True)
            callbacks.append(tensorboard_callback)
        # TensorBoard command: tensorboard --logdir logs

        # Early stopping callback
        if early_stopping_patience is not None:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=early_stopping_patience, 
                restore_best_weights=True
            )
            callbacks.append(early_stopping)

        self.history = self.fit(
            [mu, X], y, epochs=epochs, batch_size=batch_size, verbose=verbose,
            validation_data=([mu_val, X_val], y_val), validation_freq=validation_freq,
            callbacks=callbacks
        )

    ##
    def plot_training_history(self) -> None:
        """
        Plots the training and validation loss over epochs.
        This method should be called after training the model using ``train_model``.
        
        :raises ValueError: If the model has no training history.
        """
        if self.history is None:
            raise ValueError("The model has no training history. Train the model using 'train_model' method first.")

        plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
        tot_train = len(self.history.history['loss'])
        tot_valid = len(self.history.history['val_loss']) 
        valid_freq = int(tot_train / tot_valid)
        plt.plot(np.arange(tot_train), self.history.history['loss'], 'b-', label='Training loss', linewidth=2)
        plt.plot(valid_freq * np.arange(tot_valid), self.history.history['val_loss'], 'r--', label='Validation loss', linewidth=2)
        plt.yscale('log')
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.title('Training and Validation Loss', fontsize=16)
        plt.grid(True)
        plt.show()

    def summary(self, **kwargs):
        """
        Prints a summary of the DeepONet model architecture.

        :param kwargs: Additional keyword arguments to pass to the Keras summary method.
        """
        RED = "\033[91m"
        RESET = "\033[0m"

        print(f"{RED}DeepONet Model Summary:{RESET}")
        super(DeepONet, self).summary(**kwargs)

        print(f"\n\n{RED}Branch Network Summary:{RESET}")
        self.branch.summary(**kwargs)

        print(f"\n\n{RED}Trunk Network Summary:{RESET}")
        self.trunk.summary(**kwargs)