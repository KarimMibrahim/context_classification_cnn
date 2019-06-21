# General Imports
import os
import numpy as np
import pandas as pd
from time import strftime, localtime

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, TimeDistributed, Flatten, GRU, Dropout, Dense
import dzr_ml_tf.data_pipeline as dp
from dzr_ml_tf.label_processing import tf_multilabel_binarize
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

# Machine Learning preprocessing and evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, \
    hamming_loss
from sklearn.model_selection import train_test_split

#SOURCE_PATH = "/srv/workspace/research/balanceddata/"
#SPECTROGRAMS_PATH = "/srv/workspace/research/balanceddata/mel_specs"
SOURCE_PATH = "/home/karim/Documents/BalancedDatasetDeezer/"
SPECTROGRAMS_PATH = "/home/karim/Documents/BalancedDatasetDeezer/mel_specs/mel_specs/"
OUTPUT_PATH = "/home/karim/Documents/research/experiments_results"


INPUT_SHAPE = (646, 96, 1)
LABELS_LIST = ['car', 'chill', 'club', 'dance', 'gym', 'happy', 'night', 'party', 'relax', 'running',
               'sad', 'sleep', 'summer', 'work', 'workout']


def split_dataset(csv_path=os.path.join(SOURCE_PATH, "GroundTruth/ground_truth_single_label.csv"),
                  test_size=0.25, seed=0, save_csv=True,
                  train_save_path=os.path.join(SOURCE_PATH, "GroundTruth/train_ground_truth.csv"),
                  test_save_path=os.path.join(SOURCE_PATH, "GroundTruth/test_ground_truth.csv"),
                  validation_save_path = os.path.join(SOURCE_PATH, "GroundTruth/validation_ground_truth.csv")):
    groundtruth = pd.read_csv(csv_path)
    train, validation = train_test_split(groundtruth, test_size=0.1, random_state=seed)
    train, test = train_test_split(train, test_size=test_size, random_state=seed)
    if save_csv:
        pd.DataFrame.to_csv(train, train_save_path, index=False)
        pd.DataFrame.to_csv(validation, validation_save_path, index=False)
        pd.DataFrame.to_csv(test, test_save_path, index=False)
    return train, test


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
            print("Error while computing features for" +  str(song_id) + '\n')
            return np.float32(0.0), True
            # spect = spect[:,215:215+646]
        # print(spect.shape)
        return spect, False
    except Exception as err:
        print("Error while computing features for " + str(song_id) + '\n')
        return np.float32(0.0), True


def get_model():
    # Define model architecture
    model = Sequential(
        [
            InputLayer(input_shape=INPUT_SHAPE, name="input_layer"),

            # BatchNormalization()

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
    return model


def compile_model(model, loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy']):
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


# Dataset pipelines
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

    # Apply permute dimensions
    dataset = dataset.map(lambda sample: dict(sample, features=tf.transpose(sample["features"], perm=[1, 2, 0])),
                          num_parallel_calls=num_parallel_calls)

    # Filter by shape (remove badly shaped tensors)
    dataset = dataset.filter(lambda sample: dp.check_tensor_shape(sample["features"], input_shape))

    # set features shape
    dataset = dataset.map(lambda sample: dict(sample,
                                              features=dp.set_tensor_shape(sample["features"], input_shape)))

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        dataset = dataset.cache(cache_dir)

    # one hot encoding of labels
    dataset = dataset.map(lambda sample: dict(sample, binary_label=tf_multilabel_binarize(
        sample.get("label", b""), label_list_tf=tf.constant(LABELS_LIST))[0]), )

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
    test_ground_truth = pd.read_csv(os.path.join(LOADING_PATH, "test_ground_truth.csv"))
    all_ground_truth = pd.read_pickle(os.path.join(LOADING_PATH, "ground_truth_hot_vector.pkl"))
    all_ground_truth.drop("playlists_count", axis=1, inplace=True);
    all_ground_truth = all_ground_truth[all_ground_truth.song_id.isin(test_ground_truth.song_id)]
    all_ground_truth = all_ground_truth.set_index('song_id')
    all_ground_truth = all_ground_truth.loc[test_ground_truth.song_id]
    test_classes = all_ground_truth.values
    test_classes = test_classes.astype(int)

    spectrograms = np.zeros([len(test_ground_truth), 646, 96])
    songs_ID = np.zeros([len(test_ground_truth), 1])
    for idx, filename in enumerate(list(test_ground_truth.song_id)):
        try:
            spect = np.load(os.path.join(SPECTROGRAM_PATH, str(filename) + '.npz'))['feat']
        except:
            continue
        if (spect.shape == (1, 1292, 96)):
            spectrograms[idx] = spect
            songs_ID[idx] = filename
    spectrograms = np.expand_dims(spectrograms, axis=3)
    return spectrograms, test_classes


def evaluate_model(model, spectrograms, test_classes, saving_path):
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
    print("Area Under the Curve (AUC) is: " + str(auc_roc))
    # Hamming loss is the fraction of labels that are incorrectly predicted.
    hamming_error = hamming_loss(test_classes, test_pred)
    print("Hamming Loss (ratio of incorrect tags) is: " + str(hamming_error))
    with open(os.path.join(saving_path, "evaluation_results.txt"), "w") as f:
        f.write("Exact match accuracy is: " + str(accuracy) + "%\n" + "Area Under the Curve (AUC) is: " + str(auc_roc)
                + "\nHamming Loss (ratio of incorrect tags) is: " + str(hamming_error))
    print("saving prediction to disk")
    np.savetxt(os.path.join(saving_path, 'predictions.out'), test_pred_prob, delimiter=',')
    np.savetxt(os.path.join(saving_path, 'test_ground_truth_classes.txt'), test_classes, delimiter=',')
    return accuracy, auc_roc, hamming_error


def save_model(model, path):
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(path, "model.json"), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.join.path(path, "model.h5"))
    print("Saved model to disk")


def main():
    # splitting datasets
    #split_dataset()

    # Loading datasets
    training_dataset = get_training_dataset(os.path.join(SOURCE_PATH, "GroundTruth/train_ground_truth.csv"))
    val_dataset = get_validation_dataset(os.path.join(SOURCE_PATH, "GroundTruth/validation_ground_truth.csv"))

    # TODO: path
    exp_dir = os.path.join(OUTPUT_PATH,"Pipeline_CNN")
    experiment_name = strftime("%Y-%m-%d_%H-%M-%S", localtime())

    fit_config = {
        "steps_per_epoch": 1053,
        "epochs": 20,
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
        ]
    }

    # Printing the command to run tensorboard [Just to remember]
    print("Execute the following in a terminal:\n" + "tensorboard --logdir=" + os.path.join(exp_dir, experiment_name))

    model = get_model()
    compile_model(model)

    dp.safe_remove(os.path.join(OUTPUT_PATH, 'tmp/tf_cache/training/', "_0.lockfile"))
    dp.safe_remove(os.path.join(OUTPUT_PATH, 'tmp/tf_cache/validation/', "_0.lockfile"))
    model.fit(training_dataset, validation_data=val_dataset, **fit_config)

    spectrograms, test_classes = load_test_set_raw()
    accuracy, auc_roc, hamming_error = evaluate_model(model, spectrograms, test_classes,
                                                      saving_path=os.path.join(exp_dir, experiment_name))
    # save_model(model,"path/path/path")


if __name__ == "__main__":
    main()
