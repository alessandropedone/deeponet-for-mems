import os
import absl.logging
import tensorflow as tf

def run_on_device(func: callable, *args, **kwargs):
    """
    .. admonition:: Description

        This function sets up TensorFlow to run on a GPU if available, otherwise falls back to CPU. 
        It configures the necessary environment variables and TensorFlow settings to optimize GPU usage.

    :param func: The function to be executed on the selected device.
    :param args: Positional arguments to pass to the function.
    :param kwargs: Keyword arguments to pass to the function.
    :return: The result of the function execution.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 only (optional)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # "0" = all logs, "1" = filter INFO, "2" = filter INFO & WARNING, "3" = only ERRORs
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="  # Prevent CUDA libdevice warnings
    absl.logging.set_verbosity(absl.logging.ERROR)

    # List GPUs and set memory growth
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass  # Already set or not supported

    device = '/GPU:0' if gpus else '/CPU:0'

    with tf.device(device):
        # Dummy op to check placement
        a = tf.constant([[1.0]])
        if 'GPU' in a.device:
            print("✅ Using GPU for execution")
        else:
            print("⚠️ Using CPU for execution")

        # Call the provided function with arguments
        return func(*args, **kwargs)