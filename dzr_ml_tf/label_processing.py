"""
    utils functions for processing labels data
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer


def load_activity_annotation(annotation_path, n_time_samples, crop, label_mapping=None, dtype=np.float32):
    sampling_frequency = float(n_time_samples)/(crop[1]-crop[0])
    df = pd.read_csv(annotation_path)
    # if not label_mapping:
    #     label_mapping = {label:index for index,label in enumerate(set(df["label"]))}

    df["label_index"] = df.label.apply(lambda x:label_mapping[x])
    annotation_array = np.zeros((n_time_samples, len(label_mapping)), dtype=dtype)
    for _,row in df.iterrows():
        start_index = max(int((row["seg_start"] - crop[0]) * sampling_frequency),0)
        end_index = max(int((row["seg_end"] - crop[0]) * sampling_frequency),0)
        annotation_array[start_index:end_index,label_mapping[row["label"]]] = 1

    return annotation_array


def one_hot_label(label_string_tf, label_list_tf, dtype=tf.float32):
    """
        Transform string label to one hot vector.
    """
    return tf.cast(tf.equal(label_list_tf, label_string_tf), dtype)



def sklearn_mlb(tsv_string, label_list):
    mlb = MultiLabelBinarizer(classes=label_list)
    mlb.fit([[]])
    res = mlb.transform([tsv_string.split(b"\t")]).astype(np.float32).flatten()
    return res

def tf_multilabel_binarize(tf_tsv_string, label_list_tf):
    input_args = [
                    tf_tsv_string,
                    label_list_tf,
                 ]
    res = tf.py_func(sklearn_mlb,
        input_args,
        (tf.float32),
        stateful=False),
    return res