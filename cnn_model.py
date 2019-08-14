# General Imports
import os
import numpy as np
import pandas as pd
from time import strftime, localtime
import matplotlib.pyplot as plt
from utilities import create_analysis_report

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, TimeDistributed, Flatten, GRU, Dropout, Dense, \
    BatchNormalization
import dzr_ml_tf.data_pipeline as dp
from dzr_ml_tf.label_processing import tf_multilabel_binarize
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops

from focal_loss import focal_loss

# Machine Learning preprocessing and evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, \
    hamming_loss
from sklearn.model_selection import train_test_split
from dzr_ml_tf.device import limit_memory_usage
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import check_random_state

limit_memory_usage(0.3)
plt.rcParams.update({'font.size': 22})
# os.environ["CUDA_VISIBLE_DEVICES"]="2"


SOURCE_PATH = "/home/karim/Documents/research/sourceCode/context_classification_cnn/"
SPECTROGRAMS_PATH = "/home/karim/Documents/BalancedDatasetDeezer/mel_specs/mel_specs/"
OUTPUT_PATH = "/home/karim/Documents/research/experiments_results"

#SOURCE_PATH = "/srv/workspace/research/context_classification_cnn/"
#SPECTROGRAMS_PATH = "/srv/workspace/research/balanceddata/mel_specs/"
#OUTPUT_PATH = "/srv/workspace/research/balanceddata/experiments_results"


EXPERIMENTNAME = "C4_square_normalized_positive_weight_by_max"
INPUT_SHAPE = (646, 96, 1)
LABELS_LIST = ['car', 'chill', 'club', 'dance', 'gym', 'happy', 'night', 'party', 'relax', 'running',
               'sad', 'sleep', 'summer', 'work', 'workout']

#TEMPORARY VARIABLES TO SPEED UP WEIGHTED LOSS COMPUTATIONS [fix later]
global_weights_positive = pd.read_csv(os.path.join(SOURCE_PATH, "GroundTruth/positive_weights_allones.csv"))
global_weights_negative = pd.read_csv(os.path.join(SOURCE_PATH, "GroundTruth/negative_weights_allones.csv"))
global_labels = pd.read_csv(os.path.join(SOURCE_PATH, "GroundTruth/balanced_ground_truth_hot_vector.csv"))
resolution = pd.read_csv(os.path.join(SOURCE_PATH, "GroundTruth/IDs_resolution.csv"))
resolution.set_index("label")

def load_spectrogram_tf(sample, identifier_key="song_id",
                        path="/my_data/MelSpectograms_top20/", device="/cpu:0",
                        features_key="features"):
    """
        wrap load_spectrogram into a tensorflow function.
    """
    with tf.device(device):
        input_args = [sample[identifier_key], tf.constant(path)]
        res = tf.py_func(load_spectrogram,
                         input_args,
                         (tf.float32, tf.bool),
                         stateful=False),
        spectrogram, error = res[0]

        res = dict(list(sample.items()) + [(features_key, spectrogram), ("error", error)])
        return res


def load_spectrogram(*args):
    """
        loads spectrogram with error tracking.
        args : song ID, path to dataset
        return:
            Features: numpy ndarray, computed features (if no error occured, otherwise: 0)
            Error: boolean, False if no error, True if an error was raised during features computation.
    """
    # TODO: edit path
    path = SPECTROGRAMS_PATH
    song_id, dummy_path = args
    try:
        # tf.logging.info(f"Load spectrogram for {song_id}")
        spect = np.load(os.path.join(path, str(song_id) + '.npz'))['arr_0']
        if (spect.shape != (1, 646, 96)):
            # print("\n Error while computing features for" +  str(song_id) + '\n')
            return np.float32(0.0), True
            # spect = spect[:,215:215+646]
        # print(spect.shape)
        return spect, False
    except Exception as err:
        # print("\n Error while computing features for " + str(song_id) + '\n')
        return np.float32(0.0), True


def get_model():
    # Define model architecture

    # C1_freq
    """
    model = Sequential(
        [
            InputLayer(input_shape=INPUT_SHAPE, name="input_layer"),

            BatchNormalization(),

            Conv2D(activation="relu", filters=32, kernel_size=[32, 1], name="conv_1", padding="same"),
            MaxPooling2D(name="max_pool_1", padding="valid", pool_size=[1, 80]),

            Flatten(),
            Dense(200, activation='sigmoid', name="dense_1"),
            Dropout(name="dropout_1", rate=0.5),
            Dense(15, activation='sigmoid', name="dense_2"),
        ]
    )
    """

    # C1_time
    """
    model = Sequential(
        [
            InputLayer(input_shape=INPUT_SHAPE, name="input_layer"),

            BatchNormalization(),

            Conv2D(activation="relu", filters=32, kernel_size=[1, 60], name="conv_1", padding="same"),
            MaxPooling2D(name="max_pool_1", padding="valid", pool_size=[96, 1]),

            Flatten(),
            Dense(200, activation='sigmoid', name="dense_1"),
            Dropout(name="dropout_1", rate=0.5),
            Dense(15, activation='sigmoid', name="dense_2"),
        ]
    )
    """

    # C4_model

    model = Sequential(
        [
            InputLayer(input_shape=INPUT_SHAPE, name="input_layer"),

            BatchNormalization(),

            Conv2D(activation="relu", filters=32, kernel_size=[3, 3], name="conv_1", padding="same"),
            MaxPooling2D(name="max_pool_1", padding="valid", pool_size=[2, 2]),

            Conv2D(activation="relu", filters=64, kernel_size=[3, 3], name="conv_2", padding="same", use_bias=True),
            MaxPooling2D(name="max_pool_2", padding="valid", pool_size=[2, 2]),

            Conv2D(activation="relu", filters=128, kernel_size=[3, 3], name="conv_3", padding="same", use_bias=True),
            MaxPooling2D(name="max_pool_3", padding="valid", pool_size=[2, 2]),

            Conv2D(activation="relu", filters=256, kernel_size=[3, 3], name="conv_4", padding="same", use_bias=True),
            MaxPooling2D(name="max_pool_4", padding="valid", pool_size=[2, 2]),

            # TimeDistributed(layer=Flatten(name="Flatten"), name="TD_Flatten"),
            # GRU(activation="tanh", dropout=0.1, name="gru_1", recurrent_activation="hard_sigmoid", recurrent_dropout=0.1,
            #        return_sequences=False, trainable=True, units=512, use_bias=True),

            # Dropout(name="dropout_1", rate=0.3),
            # Dense(activation="sigmoid", name="dense_1", trainable=True, units=20),

            Flatten(),
            Dense(256, activation='sigmoid', name="dense_1"),
            Dropout(name="dropout_1", rate=0.3),
            Dense(15, activation='sigmoid', name="dense_2"),
        ]
    )

    # C2_model
    """
    model = Sequential(
        [
            InputLayer(input_shape=INPUT_SHAPE, name="input_layer"),

            BatchNormalization(),

            Conv2D(activation="relu", filters=32, kernel_size=[3, 3], name="conv_1", padding="same"),
            MaxPooling2D(name="max_pool_1", padding="valid", pool_size=[2, 2]),

            Conv2D(activation="relu", filters=64, kernel_size=[3, 3], name="conv_2", padding="same", use_bias=True),
            MaxPooling2D(name="max_pool_2", padding="valid", pool_size=[2, 2]),

            Flatten(),
            Dense(256, activation='sigmoid', name="dense_1"),
            Dropout(name="dropout_1", rate=0.3),
            Dense(15, activation='sigmoid', name="dense_2"),
        ]
    )
    """
    return model


def compile_model(model, loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy']):
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# Dataset pipelines
def numpy_repeat_id(song_id, label_list):
    new_id = resolution[resolution.song_id == song_id].label.values[0]
    res = np.repeat(new_id, len(label_list))
    res = res.astype(np.float64)
    return res

def tf_replace_labels_with_ID(tf_song_id, label_list_tf, device="/cpu:0"):
    with tf.device(device):
        input_args = [
                        tf_song_id,
                        label_list_tf,
                     ]
        res = tf.py_func(numpy_repeat_id,
            input_args,
            (tf.float64),
            stateful=False),
        return res

def get_training_dataset(path):
    return get_dataset(path, shuffle=True,
                       cache_dir=os.path.join(OUTPUT_PATH, "tmp/tf_cache/training/"))

def get_validation_dataset(path):
    return get_dataset(path, batch_size=32, shuffle=False,
                       random_crop=False, cache_dir=os.path.join(OUTPUT_PATH, "tmp/tf_cache/validation/"))

def get_test_dataset(path):
    return get_dataset(path, batch_size=50, shuffle=False,
                       infinite_generator=False, cache_dir=os.path.join(OUTPUT_PATH, "tmp/tf_cache/test/"))

def get_dataset(input_csv, input_shape=INPUT_SHAPE, batch_size=32, shuffle=True,
                infinite_generator=True, random_crop=False, cache_dir=os.path.join(OUTPUT_PATH, "tmp/tf_cache/"),
                num_parallel_calls=32):
    # build dataset from csv file
    dataset = dp.dataset_from_csv(input_csv)
    # Shuffle data
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100, seed=1, reshuffle_each_iteration=True)

    # compute mel spectrogram
    dataset = dataset.map(lambda sample: load_spectrogram_tf(sample), num_parallel_calls=1)

    # filter out errors
    dataset = dataset.filter(lambda sample: tf.logical_not(sample["error"]))

    # map dynamic compression
    C = 100
    dataset = dataset.map(lambda sample: dict(sample, features=tf.log(1 + C * sample["features"])),
                          num_parallel_calls=num_parallel_calls)

    # Apply permute dimensions
    dataset = dataset.map(lambda sample: dict(sample, features=tf.transpose(sample["features"], perm=[1, 2, 0])),
                          num_parallel_calls=num_parallel_calls)

    # Filter by shape (remove badly shaped tensors)
    dataset = dataset.filter(lambda sample: dp.check_tensor_shape(sample["features"], input_shape))

    # set features shape
    dataset = dataset.map(lambda sample: dict(sample,
                                              features=dp.set_tensor_shape(sample["features"], input_shape)))

    # if cache_dir:
    #    os.makedirs(cache_dir, exist_ok=True)
    #    dataset = dataset.cache(cache_dir)

    # one hot encoding of labels [REPLACED NOW BY THE SONG ID TEMPORARILY]
    """
    dataset = dataset.map(lambda sample: dict(sample, binary_label=tf_multilabel_binarize(
        sample.get("label", b""), label_list_tf=tf.constant(LABELS_LIST))[0]), )

    # set output shape
    dataset = dataset.map(lambda sample: dict(sample, binary_label=dp.set_tensor_shape(
        sample["binary_label"], (len(LABELS_LIST)))))
    """

    """
    TEMPORARY TILL SONG_ID PASSING IS REPLACED BY LABEL PASSING [FOR THE CUSTOM LOSS FUNCTION]
    """
    dataset = dataset.map(lambda sample: dict(sample, binary_label= tf_replace_labels_with_ID(
        sample.get("song_id"), label_list_tf=tf.constant(LABELS_LIST))[0]), )
    # set output shape
    dataset = dataset.map(lambda sample: dict(sample, binary_label=dp.set_tensor_shape(
        sample["binary_label"], (len(LABELS_LIST)))))

    if infinite_generator:
        # Repeat indefinitly
        dataset = dataset.repeat(count=-1)

    # Make batch
    dataset = dataset.batch(batch_size)

    # Select only features and annotation
    dataset = dataset.map(lambda sample: (sample["features"], sample["binary_label"]))

    return dataset


def load_test_set_raw(LOADING_PATH=os.path.join(SOURCE_PATH, "GroundTruth/"),
                      SPECTROGRAM_PATH=SPECTROGRAMS_PATH):
    # Loading testset groundtruth
    test_ground_truth = pd.read_csv(os.path.join(LOADING_PATH, "test_ground_truth_binarized.csv"))
    all_ground_truth = pd.read_csv(os.path.join(LOADING_PATH, "balanced_ground_truth_hot_vector.csv"))
    # all_ground_truth.drop("playlists_count", axis=1, inplace=True);
    all_ground_truth = all_ground_truth[all_ground_truth.song_id.isin(test_ground_truth.song_id)]
    all_ground_truth = all_ground_truth.set_index('song_id')
    all_ground_truth = all_ground_truth.loc[test_ground_truth.song_id]
    test_classes = all_ground_truth.values
    test_classes = test_classes.astype(int)

    spectrograms = np.zeros([len(test_ground_truth), 646, 96])
    songs_ID = np.zeros([len(test_ground_truth), 1])
    for idx, filename in enumerate(list(test_ground_truth.song_id)):
        try:
            spect = np.load(os.path.join(SPECTROGRAM_PATH, str(filename) + '.npz'))['arr_0']
        except:
            continue
        if (spect.shape == (1, 646, 96)):
            spectrograms[idx] = spect
            songs_ID[idx] = filename

    # Apply same transformation as trianing [ALWAYS DOUBLE CHECK TRAINING PARAMETERS]
    C = 100
    spectrograms = np.log(1 + C * spectrograms)

    spectrograms = np.expand_dims(spectrograms, axis=3)
    return spectrograms, test_classes


def evaluate_model(model, spectrograms, test_classes, saving_path, evaluation_file_path):
    """
    Evaluates a given model using accuracy, area under curve and hamming loss
    :param model: model to be evaluated
    :param spectrograms: the test set spectrograms as an np.array
    :param test_classes: the ground truth labels
    :return: accuracy, auc_roc, hamming_error
    """
    test_pred_prob = model.predict(spectrograms)
    test_pred = np.round(test_pred_prob)
    # Accuracy
    accuracy = 100 * accuracy_score(test_classes, test_pred)
    print("Exact match accuracy is: " + str(accuracy) + "%")
    # Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    auc_roc = roc_auc_score(test_classes, test_pred_prob)
    print("Macro Area Under the Curve (AUC) is: " + str(auc_roc))
    auc_roc_micro = roc_auc_score(test_classes, test_pred_prob, average="micro")
    print("Micro Area Under the Curve (AUC) is: " + str(auc_roc_micro))
    auc_roc_weighted = roc_auc_score(test_classes, test_pred_prob, average="weighted")
    print("Weighted Area Under the Curve (AUC) is: " + str(auc_roc_weighted))
    # Hamming loss is the fraction of labels that are incorrectly predicted.
    hamming_error = hamming_loss(test_classes, test_pred)
    print("Hamming Loss (ratio of incorrect tags) is: " + str(hamming_error))
    with open(evaluation_file_path, "w") as f:
        f.write("Exact match accuracy is: " + str(accuracy) + "%\n" + "Area Under the Curve (AUC) is: " + str(auc_roc)
                + "\nMicro AUC is:" + str(auc_roc_micro) + "\nWeighted AUC is:" + str(auc_roc_weighted)
                + "\nHamming Loss (ratio of incorrect tags) is: " + str(hamming_error))
    print("saving prediction to disk")
    np.savetxt(os.path.join(saving_path, 'predictions.out'), test_pred_prob, delimiter=',')
    np.savetxt(os.path.join(saving_path, 'test_ground_truth_classes.txt'), test_classes, delimiter=',')

    create_analysis_report(test_pred,test_classes,saving_path,LABELS_LIST)

    return accuracy, auc_roc, hamming_error

def save_model(model, path):
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(path, "model.json"), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.join.path(path, "model.h5"))
    print("Saved model to disk")

def plot_loss_acuracy(history, path):
    # Plot training & validation accuracy values
    plt.figure(figsize=(10, 10))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(path, "model_accuracy.png"))
    plt.savefig(os.path.join(path, "model_accuracy.pdf"), format='pdf')
    # plt.savefig(os.path.join(path,label + "_model_accuracy.eps"), format='eps', dpi=900)
    # Plot training & validation loss values
    plt.figure(figsize=(10, 10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(path, "model_loss.png"))
    plt.savefig(os.path.join(path, "model_loss.pdf"), format='pdf')
    # plt.savefig(os.path.join(path,label + "_model_loss.eps"), format='eps', dpi=900)

# Dataset pipelines
def get_labels_weights_py(y_true):
    #new_ids = np.zeros_like(y_true[:,0],np.int64)
    #for idx, y in enumerate(y_true[:, 0]):
    #    new_ids[idx] = int(resolution[resolution.label == y].song_id.values[0])
    new_ids = resolution.loc[y_true[:, 0]].song_id.values
    labels = global_labels[global_labels.song_id.isin(new_ids)]
    sample_label = labels.iloc[:, 1:].values
    weights_positive = global_weights_positive[global_weights_positive.song_id.isin(new_ids)]
    samples_weights_positive = weights_positive.iloc[:,1:].values
    weights_negative = global_weights_negative[global_weights_negative.song_id .isin(new_ids)]
    samples_weights_negative = weights_negative.iloc[:,1:].values
    sample_label = sample_label.astype(np.float32)
    samples_weights_positive = samples_weights_positive.astype(np.float32)
    samples_weights_negative = samples_weights_negative.astype(np.float32)
    return sample_label, samples_weights_positive, samples_weights_negative

def tf_get_labels_weights_py(y_true,device = "/cpu:0"):
    with tf.device(device):
        input_args = [y_true]
        res = tf.py_func(get_labels_weights_py,
            input_args,
            [tf.float32, tf.float32, tf.float32],
            stateful=False)
        return res

def custom_loss(y_true, y_pred):
    labels, weights_positive, weights_negative = tf_get_labels_weights_py(y_true)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = (-labels * K.log(y_pred) * weights_positive) - ((1.0 - labels) * K.log(1.0 - y_pred) * weights_negative)
    #loss = K.mean(loss)
    return loss

def custom_loss_no_weights(y_true, y_pred):
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = (-y_true * K.log(y_pred)) - ((1.0 - y_true) * K.log(1.0 - y_pred))
    #loss = K.mean(loss)
    return loss

def originalCrossEntropymetric(y_true, y_pred):
    labels, weights_positive, weights_negative = tf_get_labels_weights_py(y_true)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    logits = tf.log(y_pred / (1 - y_pred))
    tfLoss = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits=logits)
    #tfLoss = tf.math.reduce_mean(tfLoss)
    return tfLoss

def negative_weighted_loss(y_true, y_pred):
    labels, weights_positive, weights_negative = tf_get_labels_weights_py(y_true)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = -((1.0 - labels) * K.log(1.0 - y_pred) * weights_negative)
    loss = K.mean(loss)
    return loss

def positive_weighted_loss(y_true, y_pred):
    labels, weights_positive, weights_negative = tf_get_labels_weights_py(y_true)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = (-labels * K.log(y_pred) * weights_positive)
    loss = K.mean(loss)
    return loss


def sigmoid_cross_entropy_with_logits(  # pylint: disable=invalid-name
    _sentinel=None,
    labels=None,
    logits=None,
    name=None):
  """Computes sigmoid cross entropy given `logits`.
  Measures the probability error in discrete classification tasks in which each
  class is independent and not mutually exclusive.  For instance, one could
  perform multilabel classification where a picture can contain both an elephant
  and a dog at the same time.
  For brevity, let `x = logits`, `z = labels`.  The logistic loss is
        z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      = (1 - z) * x + log(1 + exp(-x))
      = x - x * z + log(1 + exp(-x))
  For x < 0, to avoid overflow in exp(-x), we reformulate the above
        x - x * z + log(1 + exp(-x))
      = log(exp(x)) - x * z + log(1 + exp(-x))
      = - x * z + log(1 + exp(x))
  Hence, to ensure stability and avoid overflow, the implementation uses this
  equivalent formulation
      max(x, 0) - x * z + log(1 + exp(-abs(x)))
  `logits` and `labels` must have the same type and shape.
  Args:
    _sentinel: Used to prevent positional parameters. Internal, do not use.
    labels: A `Tensor` of the same type and shape as `logits`.
    logits: A `Tensor` of type `float32` or `float64`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` of the same shape as `logits` with the componentwise
    logistic losses.
  Raises:
    ValueError: If `logits` and `labels` do not have the same shape.
  """
  # pylint: disable=protected-access
  nn_ops._ensure_xent_args("sigmoid_cross_entropy_with_logits", _sentinel,
                           labels, logits)
  # pylint: enable=protected-access

  with ops.name_scope(name, "logistic_loss", [logits, labels]) as name:
    logits = ops.convert_to_tensor(logits, name="logits")
    labels = ops.convert_to_tensor(labels, name="labels")
    try:
      labels.get_shape().merge_with(logits.get_shape())
    except ValueError:
      raise ValueError("logits and labels must have the same shape (%s vs %s)" %
                       (logits.get_shape(), labels.get_shape()))

    # The logistic loss formula from above is
    #   x - x * z + log(1 + exp(-x))
    # For x < 0, a more numerically stable formula is
    #   -x * z + log(1 + exp(x))
    # Note that these two expressions can be combined into the following:
    #   max(x, 0) - x * z + log(1 + exp(-abs(x)))
    # To allow computing gradients at zero, we define custom versions of max and
    # abs functions.
    zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
    cond = (logits >= zeros)
    relu_logits = array_ops.where(cond, logits, zeros)
    neg_abs_logits = array_ops.where(cond, -logits, logits)
    return math_ops.add(
        relu_logits - logits * labels,
        math_ops.log1p(math_ops.exp(neg_abs_logits)),
        name=name)

def main():
    # splitting datasets
    # split_dataset()

    # Loading datasets
    training_dataset = get_training_dataset(os.path.join(SOURCE_PATH, "GroundTruth/train_ground_truth.csv"))
    val_dataset = get_validation_dataset(os.path.join(SOURCE_PATH, "GroundTruth/validation_ground_truth.csv"))
    positive_weights = pd.read_csv(os.path.join(SOURCE_PATH, "GroundTruth/positive_weights.csv"))
    negative_weights = pd.read_csv(os.path.join(SOURCE_PATH, "GroundTruth/negative_weights.csv"))


    # TODO: path
    exp_dir = os.path.join(OUTPUT_PATH, EXPERIMENTNAME)
    experiment_name = strftime("%Y-%m-%d_%H-%M-%S", localtime())

    fit_config = {
        "steps_per_epoch": 1053,
        "epochs": 100,
        "initial_epoch": 0,
        "validation_steps": 156,
        "callbacks": [
            TensorBoard(log_dir=os.path.join(exp_dir, experiment_name)),
            ModelCheckpoint(os.path.join(exp_dir, experiment_name, "last_iter.h5"),
                            save_weights_only=False),
            ModelCheckpoint(os.path.join(exp_dir, experiment_name, "best_eval.h5"),
                            save_best_only=True,
                            monitor="val_loss",
                            save_weights_only=False)
            #,EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min')
        ]
    }

    # Printing the command to run tensorboard [Just to remember]
    print("Execute the following in a terminal:\n" + "tensorboard --logdir=" + os.path.join(exp_dir, experiment_name))

    optimization = tf.keras.optimizers.Adadelta(lr=0.01)
    model = get_model()
    loss = custom_loss
    compile_model(model, loss = loss,  optimizer=optimization, metrics=[originalCrossEntropymetric,positive_weighted_loss,negative_weighted_loss])

    dp.safe_remove(os.path.join(OUTPUT_PATH, 'tmp/tf_cache/'))
    history = model.fit(training_dataset, validation_data=val_dataset, **fit_config)

    # save model architecture to disk
    with open(os.path.join(exp_dir, experiment_name, "model_summary.txt"), 'w+') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    # Load model with best validation results and apply on testset
    model.load_weights(os.path.join(exp_dir, experiment_name, "best_eval.h5"))
    spectrograms, test_classes = load_test_set_raw()
    accuracy, auc_roc, hamming_error = evaluate_model(model, spectrograms, test_classes,
                                                      saving_path=os.path.join(exp_dir, experiment_name),
                                                      evaluation_file_path=os.path.join(exp_dir, experiment_name,
                                                                                        "evaluation_results.txt"))

    # save_model(model,"path/path/path")
    plot_loss_acuracy(history, os.path.join(exp_dir, experiment_name))

    # Evaluate on old dataset
    # old_specs, old_test_classes = load_old_test_set_raw()
    # print("\nEvaluating on old testset:")
    # accuracy, auc_roc, hamming_error = evaluate_model(model, old_specs, old_test_classes,saving_path= os.path.join(exp_dir, experiment_name),
    #                                                 evaluation_file_path=os.path.join(exp_dir, experiment_name, "old_evaluation_results.txt"))


if __name__ == "__main__":
    main()
