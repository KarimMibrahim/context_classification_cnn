# General Imports
import os
import numpy as np
import pandas as pd
from time import strftime, localtime
import matplotlib.pyplot as plt

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, TimeDistributed, Flatten, GRU, Dropout, Dense, \
    BatchNormalization
import dzr_ml_tf.data_pipeline as dp
from dzr_ml_tf.label_processing import tf_multilabel_binarize
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras import backend as K

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

# SOURCE_PATH = "/srv/workspace/research/context_classification_cnn/"
# SPECTROGRAMS_PATH = "/srv/workspace/research/balanceddata/mel_specs/"
# OUTPUT_PATH = "/srv/workspace/research/balanceddata/experiments_results"


EXPERIMENTNAME = "C4_square"
INPUT_SHAPE = (646, 96, 1)
LABELS_LIST = ['car', 'chill', 'club', 'dance', 'gym', 'happy', 'night', 'party', 'relax', 'running',
               'sad', 'sleep', 'summer', 'work', 'workout']


def mark_groups_for_samples(df, n_samples, extra_criterion):
    """
        Return groups, an array of size n_samples, marking the group to which each sample belongs
        The default group is -1 if extra_criterion is None
        If a criterion is given (artist or album), then this information is taken into account
    """
    groups = np.array([-1 for _ in range(n_samples)])
    if extra_criterion is None:
        return groups

    if extra_criterion == "artist":
        crit_col = "artist_id"
    elif extra_criterion == "album":
        crit_col = "releasegroupmbid"
    else:
        return groups

    gp = df.groupby(crit_col)
    i_key = 0
    for g_key in gp.groups:
        samples_idx_per_group = gp.groups[g_key].tolist()
        groups[samples_idx_per_group] = i_key
        i_key += 1
    return groups


def select_fold(index_label, desired_samples_per_label_per_fold, desired_samples_per_fold, random_state):
    """
        For a label, finds the fold where the next sample should be distributed
    """
    # Find the folds with the largest number of desired samples for this label
    largest_desired_label_samples = max(desired_samples_per_label_per_fold[:, index_label])
    folds_targeted = np.where(desired_samples_per_label_per_fold[:, index_label] == largest_desired_label_samples)[0]

    if len(folds_targeted) == 1:
        selected_fold = folds_targeted[0]
    else:
        # Break ties by considering the largest number of desired samples
        largest_desired_samples = max(desired_samples_per_fold[folds_targeted])
        folds_re_targeted = np.intersect1d(np.where(
            desired_samples_per_fold == largest_desired_samples)[0], folds_targeted)

        # If there is still a tie break it picking a random index
        if len(folds_re_targeted) == 1:
            selected_fold = folds_re_targeted[0]
        else:
            selected_fold = random_state.choice(folds_re_targeted)
    return selected_fold


def iterative_split(df, out_file, target, n_splits, extra_criterion=None, seed=None):
    """
        Implement iterative split algorithm
        df is the input data
        out_file is the output file containing the same data as the input plus a column about the fold
        n_splits the number of folds
        target is the target source for which the files are generated
        extra_criterion, an extra condition to be taken into account in the split such as the artist
    """
    print("Starting the iterative split")
    random_state = check_random_state(seed)

    mlb_target = MultiLabelBinarizer()
    M = mlb_target.fit_transform(df[target].str.split('\t'))

    n_samples = len(df)
    n_labels = len(mlb_target.classes_)

    # If the extra criterion is given create "groups", which shows to which group each sample belongs
    groups = mark_groups_for_samples(df, n_samples, extra_criterion)

    ratios = np.ones((1, n_splits)) / n_splits
    # Calculate the desired number of samples for each fold
    desired_samples_per_fold = ratios.T * n_samples

    # Calculate the desired number of samples of each label for each fold
    number_samples_per_label = np.asarray(M.sum(axis=0)).reshape((n_labels, 1))
    desired_samples_per_label_per_fold = np.dot(ratios.T, number_samples_per_label.T)  # shape: n_splits, n_samples

    seen = set()
    out_folds = np.array([-1 for _ in range(n_samples)])

    count_seen = 0
    print("Going through the samples")
    while n_samples > 0:
        # Find the index of the label with the fewest remaining examples
        valid_idx = np.where(number_samples_per_label > 0)[0]
        index_label = valid_idx[number_samples_per_label[valid_idx].argmin()]
        label = mlb_target.classes_[index_label]

        # Find the samples belonging to the label with the fewest remaining examples
        # second select all samples belonging to the selected label and remove the indices
        # of the samples which have been already seen
        all_label_indices = set(M[:, index_label].nonzero()[0])
        indices = all_label_indices - seen
        assert (len(indices) > 0)

        print(label, index_label, number_samples_per_label[index_label], len(indices))

        for i in indices:
            if i in seen:
                continue

            # Find the folds with the largest number of desired samples for this label
            selected_fold = select_fold(index_label, desired_samples_per_label_per_fold,
                                        desired_samples_per_fold, random_state)

            # put in this fold all the samples which belong to the same group
            idx_same_group = np.array([i])
            if groups[i] != -1:
                idx_same_group = np.where(groups == groups[i])[0]

            # Update the folds, the seen, the number of samples and desired_samples_per_fold
            out_folds[idx_same_group] = selected_fold
            seen.update(idx_same_group)
            count_seen += idx_same_group.size
            n_samples -= idx_same_group.size
            desired_samples_per_fold[selected_fold] -= idx_same_group.size

            # The sample may have multiple labels so update for all
            for idx in idx_same_group:
                all_labels = M[idx].nonzero()
                desired_samples_per_label_per_fold[selected_fold, all_labels] -= 1
                number_samples_per_label[all_labels] -= 1

    df['fold'] = out_folds
    df.drop("index", axis=1, inplace=True)
    print(count_seen, len(df))
    df.to_csv(out_file, sep=',', index=False)
    return df


def split_dataset(csv_path=os.path.join(SOURCE_PATH, "GroundTruth/ground_truth_single_label.csv"),
                  artists_csv_path=os.path.join(SOURCE_PATH, "GroundTruth/songs_artists.tsv"),
                  test_size=0.25, seed=0, save_csv=True, n_splits=4,
                  train_save_path=os.path.join(SOURCE_PATH, "GroundTruth/"),
                  test_save_path=os.path.join(SOURCE_PATH, "GroundTruth/"),
                  validation_save_path=os.path.join(SOURCE_PATH, "GroundTruth/"),
                  folds_save_path=os.path.join(SOURCE_PATH, "GroundTruth/ground_truth_folds.csv")):
    song_artist = pd.read_csv(artists_csv_path, delimiter='\t')
    groundtruth = pd.read_csv(csv_path)
    ground_truth_artist = groundtruth.merge(song_artist, on='song_id')
    ground_truth_artist = ground_truth_artist.drop_duplicates("song_id")
    ground_truth_artist = ground_truth_artist.reset_index()

    groundtruth_folds = iterative_split(df=ground_truth_artist, out_file=folds_save_path, target='label',
                                        n_splits=n_splits, extra_criterion='artist', seed=seed)
    test = groundtruth_folds[groundtruth_folds.fold == 0]
    train_validation_combined = groundtruth_folds[groundtruth_folds.fold.isin(np.arange(1, n_splits))]
    train, validation = train_test_split(train_validation_combined, test_size=0.1, random_state=seed)
    train.drop(["artist_id", "fold"], axis=1, inplace=True)
    test.drop(["artist_id", "fold"], axis=1, inplace=True)
    validation.drop(["artist_id", "fold"], axis=1, inplace=True)
    # train, test = train_test_split(train, test_size=test_size, random_state=seed)
    if save_csv:
        pd.DataFrame.to_csv(train, os.path.join(train_save_path, "train_ground_truth.csv"), index=False)
        pd.DataFrame.to_csv(validation, os.path.join(validation_save_path, "validation_ground_truth.csv"), index=False)
        pd.DataFrame.to_csv(test, os.path.join(test_save_path, "test_ground_truth.csv"), index=False)
    # Save data in binarized format as well
    mlb_target = MultiLabelBinarizer()
    M = mlb_target.fit_transform(test.label.str.split('\t'))
    Mdf = pd.DataFrame(M, columns=LABELS_LIST)
    test.reset_index(inplace=True, drop=True)
    test_binarized = pd.concat([test, Mdf], axis=1)
    test_binarized.drop(['label'], inplace=True, axis=1)
    # For validation
    mlb_target = MultiLabelBinarizer()
    M = mlb_target.fit_transform(validation.label.str.split('\t'))
    Mdf = pd.DataFrame(M, columns=LABELS_LIST)
    validation.reset_index(inplace=True, drop=True)
    validation_binarized = pd.concat([validation, Mdf], axis=1)
    validation_binarized.drop(['label'], inplace=True, axis=1)
    # for training
    mlb_target = MultiLabelBinarizer()
    M = mlb_target.fit_transform(train.label.str.split('\t'))
    Mdf = pd.DataFrame(M, columns=LABELS_LIST)
    train.reset_index(inplace=True, drop=True)
    train_binarized = pd.concat([train, Mdf], axis=1)
    train_binarized.drop(['label'], inplace=True, axis=1)
    if save_csv:
        pd.DataFrame.to_csv(test_binarized, os.path.join(test_save_path, "test_ground_truth_binarized.csv"),
                            index=False)
        pd.DataFrame.to_csv(validation_binarized,
                            os.path.join(validation_save_path, "validation_ground_truth_binarized.csv"), index=False)
        pd.DataFrame.to_csv(train_binarized, os.path.join(train_save_path, "train_ground_truth_binarized.csv"),
                            index=False)
    return train, validation, test


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


def load_old_test_set_raw(LOADING_PATH=os.path.join(SOURCE_PATH, "GroundTruth/"),
                          SPECTROGRAM_PATH="/home/karim/Documents/MelSpectograms_top20/"):
    # Loading testset groundtruth
    test_ground_truth = pd.read_csv(os.path.join(LOADING_PATH, "old_test_ground_truth[unbalanced].csv"))
    all_ground_truth = pd.read_pickle(os.path.join(LOADING_PATH, "old_ground_truth_hot_vector[unblanced].pkl"))
    all_ground_truth.drop(['playlists_count', 'train', 'shower', 'park', 'morning', 'training'], axis=1, inplace=True);
    all_ground_truth = all_ground_truth[all_ground_truth.song_id.isin(test_ground_truth.song_id)]
    test_ground_truth = test_ground_truth[test_ground_truth.song_id.isin(all_ground_truth.song_id)]
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
            spect = spect[:, 323: 323 + 646, :]
            spectrograms[idx] = spect
            songs_ID[idx] = filename
    spectrograms = np.expand_dims(spectrograms, axis=3)
    return spectrograms, test_classes


def load_validation_set_raw(LOADING_PATH=os.path.join(SOURCE_PATH, "GroundTruth/"),
                            SPECTROGRAM_PATH=SPECTROGRAMS_PATH):
    # Loading testset groundtruth
    test_ground_truth = pd.read_csv(os.path.join(LOADING_PATH, "validation_ground_truth.csv"))
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


def tf_idf(track_count,hot_encoded, number_of_classes = 15):
    class_total_tracks = track_count.sum()
    class_count_per_sample = hot_encoded.iloc[:, 1:].sum(axis=1)
    track_tf = track_count.copy()
    # compute tf (number of track occurances in a context / total number of occurances in this context)
    track_tf.iloc[:, 1:] = track_count.iloc[:, 1:].div(class_total_tracks, axis=1)
    # compute idf (number of contexts / number of positive context in this class)
    track_idf = np.log(number_of_classes / class_count_per_sample)
    track_tf_idf = track_tf.copy()
    track_tf_idf.iloc[:,1:] = track_tf.iloc[:, 1:].mul(track_idf, axis=0)
    #track_tf_idf.to_csv("/home/karim/Documents/BalancedDatasetDeezer/GroundTruth/positive_weights.csv",index=False)
    return track_tf_idf

def negative_labeles_probabilities(hot_encoded):
    # count the number of times a combination has appeared with the negative label as 1 / the total number of
    # occurances of that combination without the negative label
    negative_weights = np.ones([len(hot_encoded),len(LABELS_LIST)])
    for sample_idx in range(len(hot_encoded)):
        for label_idx in range(len(LABELS_LIST)):
            if hot_encoded.iloc[sample_idx, label_idx+1] == 1:
                negative_weights[sample_idx, label_idx] = 1
            else:
                temp_combination = hot_encoded.iloc[sample_idx,1:].copy()
                temp_combination[label_idx] = 1
                positive_samples = len(hot_encoded[(hot_encoded.iloc[:, 1:].values == temp_combination.values).all(axis = 1)])
                negative_samples = len(hot_encoded[(hot_encoded.iloc[:, 1:].values == hot_encoded.iloc[sample_idx, 1:].values).all(axis=1)])
                negative_weights[sample_idx, label_idx] = positive_samples / (positive_samples + negative_samples)
    negative_weights_df = pd.DataFrame(negative_weights, columns=LABELS_LIST)
    negative_weights_df["song_id"] = hot_encoded.song_id
    negative_weights_df = negative_weights_df[["song_id"] + LABELS_LIST]
    #negative_weights_df.to_csv("/home/karim/Documents/BalancedDatasetDeezer/GroundTruth/negative_weights.csv",index=False)
    return negative_weights_df


def weighted_categorical_crossentropy(weights_positive, weights_negative):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,n) where C is the number of classes and n number of samples

    Usage:
        weights = np.array([0.5,2,10], [0.3, 1, 5]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x
        # for the first sample, etc...
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights_positive = K.variable(weights_positive)
    weights_negative = K.variable(weights_negative)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = (-y_true * K.log(y_pred) * weights_positive) - (1.0 - y_true) * (K.log(1.0 - y_pred) * weights_negative)
        #loss = y_true * K.log(y_pred) * weights_positive
        loss = K.sum(loss, -1)
        return loss

    return loss

def main():
    # splitting datasets
    # split_dataset()

    # Loading datasets
    training_dataset = get_training_dataset(os.path.join(SOURCE_PATH, "GroundTruth/train_ground_truth.csv"))
    val_dataset = get_validation_dataset(os.path.join(SOURCE_PATH, "GroundTruth/validation_ground_truth.csv"))
    positive_weights = pd.read_csv(os.path.join(SOURCE_PATH, "GroundTruth/positive_weights.csv"))
    negative_weights = pd.read_csv(os.path.join(negative_weights, "GroundTruth/positive_weights.csv"))


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
                            save_weights_only=False),
            EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min')
        ]
    }

    # Printing the command to run tensorboard [Just to remember]
    print("Execute the following in a terminal:\n" + "tensorboard --logdir=" + os.path.join(exp_dir, experiment_name))

    optimization = tf.keras.optimizers.Adadelta(lr=0.01)
    model = get_model()
    loss = weighted_categorical_crossentropy(positive_weights,negative_weights)
    compile_model(model, loss= loss,  optimizer=optimization)

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
