import tensorflow as tf
from tensorflow.keras.backend import set_session

def limit_memory_usage(gpu_memory_fraction=0.1):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    set_session(tf.Session(config=config))