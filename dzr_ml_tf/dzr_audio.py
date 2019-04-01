"""
    utils functions for using dzr_audio with Tensorflow
"""
import json

import numpy as np
import tensorflow as tf

from dzr_audio.transforms.features_extractor import compute_features


def safe_compute_features(*args):
    """
        compute features with error tracking.
        args : same as dzr_audio.transforms.features_extractor.compute_features

        return:
            Features: numpy ndarray, computed features (if no error occured, otherwise: 0)
            Error: boolean, False if no error, True if an error was raised during features computation.
    """
    song_id, entity_type, crop, transform_config, features_config = args
    try:
        tf.logging.info(f"Compute features for {song_id} on segment {crop}.")
        features,_ = compute_features(song_id, entity_type, crop, transform_config, features_config)
        return features, False

    except Exception as err:
        tf.logging.warn(f"Error while computing features for {song_id} on segment {crop}: {err}")
        return np.float32(0.0), True



def compute_features_tf(sample, features_config="{}", identifier_key="sng_id", entity_type="song", seg_start_key="seg_start", seg_duration_key="seg_duration", features_key="features", device="/cpu:0"):
    """
        wrap compute_features into a tensorflow function.
    """
    with tf.device(device):
        input_args = [
                        sample[identifier_key],
                        tf.constant(entity_type),
                        (tf.cast(sample[seg_start_key],tf.float64), tf.cast(sample[seg_start_key],tf.float64)+tf.cast(sample[seg_duration_key],tf.float64)),
                        tf.constant("{}"),
                        tf.constant(json.dumps(features_config))
                     ]
        res = tf.py_func(safe_compute_features,
                input_args,
                (tf.float32, tf.bool),
                stateful=False),
        features, error = res[0]

        res = dict(list(sample.items()) + [(features_key,features), ("error",error)])
        return res
