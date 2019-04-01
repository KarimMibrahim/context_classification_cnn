"""
    utility functions for loading and processing data
"""
import os

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MultiLabelBinarizer


def dataset_from_csv(csv_path, **kwargs):
    """
        Load dataset from a csv file.
        kwargs are forwarded to the pandas.read_csv function.
    """
    df = pd.read_csv(csv_path, **kwargs)

    dataset = (
        tf.data.Dataset.from_tensor_slices(
            {
                key:df[key].values
                for key in df
            }
        )
    )
    return dataset


def one_sample_dataset(**kwargs):
    return tf.data.Dataset.from_tensor_slices(
            {k:[v] for k,v in kwargs.items()}
        )

def safe_remove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass


def check_tensor_shape(tensor_tf, target_shape):
    """
        Return a Tensorflow boolean graph that indicates whether sample[features_key] has the specified target shape
        Only check not None entries of target_shape.
    """
    res = tf.constant(True)
    for idx,target_length in enumerate(target_shape):
        if target_length:
            res = tf.logical_and(res, tf.equal(tf.constant(target_length), tf.shape(tensor_tf)[idx]))

    return res

def set_tensor_shape(tensor, tensor_shape):
        """
            set shape for a tensor (not in place, as opposed to tf.set_shape)
        """
        tensor.set_shape(tensor_shape)
        return tensor


def crop_tensor_beforelastdim(tensor, crop):
    return tensor[...,crop[0]:crop[1],:]



def sklearn_mlb(tsv_string, label_list):
    mlb = MultiLabelBinarizer(classes=label_list)
    mlb.fit([[]])
    res = mlb.transform([tsv_string.split(b"\t")]).astype(np.float32).flatten()
    return res

def tf_multilabel(tf_tsv_string, label_list):
    input_args = [
                    tf_tsv_string,
                    tf.constant(label_list),
                 ]
    res = tf.py_func(sklearn_mlb,
            input_args,
            (tf.float32),
            stateful=False),
    return res