"""
A file to summarize the specified surrogate model.

Example usage::
    
    python -m surrogate.summary --model_path "models/potential1.keras"
"""



def main():
    # Read input arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/model.keras", help="Path to save the trained model.")

    args = parser.parse_args()
    model_path = args.model_path

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


if __name__ == "__main__":
    main()