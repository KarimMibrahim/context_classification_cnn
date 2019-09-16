# General Imports
import os
import numpy as np
import pandas as pd
from time import strftime, localtime
import matplotlib.pyplot as plt
from utilities import create_analysis_report, load_validation_set_raw

# Deep Learning
import tensorflow as tf

import dzr_ml_tf.data_pipeline as dp
from dzr_ml_tf.label_processing import tf_multilabel_binarize

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
OUTPUT_PATH = "/home/karim/Documents/research/experiments_results/"


SOURCE_PATH = "/srv/workspace/research/context_classification_cnn/"
SPECTROGRAMS_PATH = "/srv/workspace/research/balanceddata/mel_specs/"
OUTPUT_PATH = "/srv/workspace/research/balanceddata/experiments_results/"


EXPERIMENTNAME = "tf_C4_training_on_my_loss_both_weights_threshold_optimized"
INPUT_SHAPE = (646, 96, 1)
LABELS_LIST = ['car', 'chill', 'club', 'dance', 'gym', 'happy', 'night', 'party', 'relax', 'running',
               'sad', 'sleep', 'summer', 'work', 'workout']

global_weights_positive = pd.read_csv(os.path.join(SOURCE_PATH, "GroundTruth/positive_weights_MeanOne_unitVar_clip2.csv"))
global_weights_negative = pd.read_csv(os.path.join(SOURCE_PATH, "GroundTruth/negative_weights.csv"))



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

# Dataset pipelines
def get_weights_py(sample_song_id):
    weights_positive = global_weights_positive[global_weights_positive.song_id == sample_song_id]
    samples_weights_positive = weights_positive.iloc[:, 1:].values.flatten()
    weights_negative = global_weights_negative[global_weights_negative.song_id == sample_song_id]
    samples_weights_negative = weights_negative.iloc[:, 1:].values.flatten()
    samples_weights_positive = samples_weights_positive.astype(np.float32)
    samples_weights_negative = samples_weights_negative.astype(np.float32)
    #print(samples_weights_positive)
    return samples_weights_positive, samples_weights_negative

def tf_get_weights_py(sample,device = "/cpu:0"):
    with tf.device(device):
        input_args = [sample["song_id"]]
        positive_weights, negative_weights = tf.py_func(get_weights_py,
            input_args,
            [tf.float32, tf.float32],
            stateful=False)
        res = dict(list(sample.items()) + [("positive_weights", positive_weights), ("negative_weights", negative_weights)])
        return res

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

    dataset = dataset.map(lambda sample: dict(sample, binary_label=tf_multilabel_binarize(
        sample.get("label", b""), label_list_tf=tf.constant(LABELS_LIST))[0]), )

    # set output shape
    dataset = dataset.map(lambda sample: dict(sample, binary_label=dp.set_tensor_shape(
        sample["binary_label"], (len(LABELS_LIST)))))

    # load weights
    dataset = dataset.map(lambda sample: tf_get_weights_py(sample), num_parallel_calls=1)
    # set weights shape
    dataset = dataset.map(lambda sample: dict(sample, positive_weights=dp.set_tensor_shape(
        sample["positive_weights"], (len(LABELS_LIST)))))
    dataset = dataset.map(lambda sample: dict(sample, negative_weights=dp.set_tensor_shape(
        sample["negative_weights"], (len(LABELS_LIST)))))

    if infinite_generator:
        # Repeat indefinitly
        dataset = dataset.repeat(count=-1)

    # Make batch
    dataset = dataset.batch(batch_size)

    # Select only features and annotation
    dataset = dataset.map(lambda sample: (sample["features"], sample["binary_label"], sample["positive_weights"], sample["negative_weights"]))

    return dataset


def get_training_dataset(path):
    return get_dataset(path, shuffle=True,
                       cache_dir=os.path.join(OUTPUT_PATH, "tmp/tf_cache/training/"))


def get_validation_dataset(path):
    return get_dataset(path, batch_size=32, shuffle=False,
                       random_crop=False, cache_dir=os.path.join(OUTPUT_PATH, "tmp/tf_cache/validation/"))


def get_test_dataset(path):
    return get_dataset(path, batch_size=50, shuffle=False,
                       infinite_generator=False, cache_dir=os.path.join(OUTPUT_PATH, "tmp/tf_cache/test/"))


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


def get_weights(shape):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    #variable_summaries(w)
    return w


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    b = tf.Variable(initial)
    #variable_summaries(b)
    return b


def conv_2d(x, W, name=""):
    return tf.nn.conv2d(x, W, [1, 1, 1, 1], padding="SAME", name=name)


def max_pooling(x, shape, name=""):
    return tf.nn.max_pool(x, shape, strides=[1, 2, 2, 1], padding="SAME", name=name)


def conv_layer_with_reul(input, shape, name=""):
    W = get_weights(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv_2d(input, W, name) + b)


def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = get_weights([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b


def get_model(x_input, current_keep_prob, train_phase):
    # Define model architecture
    # C4_model
    x_norm = tf.layers.batch_normalization(x_input, training=train_phase)

    with tf.name_scope('CNN_1'):
        conv1 = conv_layer_with_reul(x_norm, [3, 3, 1, 32], name="conv_1")
        max1 = max_pooling(conv1, shape=[1, 2, 2, 1], name="max_pool_1")

    with tf.name_scope('CNN_2'):
        conv2 = conv_layer_with_reul(max1, [3, 3, 32, 64], name="conv_2")
        max2 = max_pooling(conv2, shape=[1, 2, 2, 1], name="max_pool_2")

    with tf.name_scope('CNN_3'):
        conv3 = conv_layer_with_reul(max2, [3, 3, 64, 128], name="conv_3")
        max3 = max_pooling(conv3, shape=[1, 2, 2, 1], name="max_pool_3")

    with tf.name_scope('CNN_4'):
        conv4 = conv_layer_with_reul(max3, [3, 3, 128, 256], name="conv_4")
        max4 = max_pooling(conv4, shape=[1, 2, 2, 1], name="max_pool_4")

    with tf.name_scope('CNN_5'):
        conv5 = conv_layer_with_reul(max4, [3, 3, 128, 256], name="conv_5")
        max5 = max_pooling(conv5, shape=[1, 2, 2, 1], name="max_pool_5")

    with tf.name_scope('CNN_6'):
        conv6 = conv_layer_with_reul(max5, [3, 3, 128, 256], name="conv_6")
        max6 = max_pooling(conv6, shape=[1, 2, 2, 1], name="max_pool_6")

    with tf.name_scope('Fully_connected_1'):
        flattened = tf.reshape(max6, [-1, 41 * 6 * 256])
        fully1 = tf.nn.sigmoid(full_layer(flattened, 256))

    with tf.name_scope('Fully_connected_2'):
        dropped = tf.nn.dropout(fully1, keep_prob=current_keep_prob)
        logits = full_layer(dropped, 15)

    output = tf.nn.sigmoid(logits)
    tf.summary.histogram('outputs', output)
    return logits, output


def evaluate_model(test_pred_prob, test_classes, saving_path, evaluation_file_path):
    """
    Evaluates a given model using accuracy, area under curve and hamming loss
    :param model: model to be evaluated
    :param spectrograms: the test set spectrograms as an np.array
    :param test_classes: the ground truth labels
    :return: accuracy, auc_roc, hamming_error
    """
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


def plot_loss_acuracy(epoch_losses_history, epoch_accurcies_history, val_losses_history, val_accuracies_history, path):
    # Plot training & validation accuracy values
    plt.figure(figsize=(10, 10))
    plt.plot(epoch_accurcies_history)
    plt.plot(val_accuracies_history)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(path, "model_accuracy.png"))
    plt.savefig(os.path.join(path, "model_accuracy.pdf"), format='pdf')
    # plt.savefig(os.path.join(path,label + "_model_accuracy.eps"), format='eps', dpi=900)
    # Plot training & validation loss values
    plt.figure(figsize=(10, 10))
    plt.plot(epoch_losses_history)
    plt.plot(val_losses_history)
    plt.title('Model loss (Cross Entropy without weighting)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(path, "model_loss.png"))
    plt.savefig(os.path.join(path, "model_loss.pdf"), format='pdf')
    # plt.savefig(os.path.join(path,label + "_model_loss.eps"), format='eps', dpi=900)


def plot_new_old_loss(epoch_losses_history, new_loss_history, path):
    # Plot training & validation accuracy values
    plt.figure(figsize=(10, 10))
    # Plot training new and old loss values
    plt.figure(figsize=(10, 10))
    plt.plot(epoch_losses_history)
    plt.plot(new_loss_history)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Origina Cross Entropy', 'Weighted Cross Entropy'], loc='upper left')
    plt.savefig(os.path.join(path, "loss_comparison.png"))
    plt.savefig(os.path.join(path, "loss_comparison.pdf"), format='pdf')


def plot_my_loss_validation(my_loss_history, my_loss_val_history, path):
    # Plot training & validation accuracy values
    plt.figure(figsize=(10, 10))
    # Plot training new and old loss values
    plt.figure(figsize=(10, 10))
    plt.plot(my_loss_history)
    plt.plot(my_loss_val_history)
    plt.title('Weighted model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(path, "weighted_loss_validation.png"))
    plt.savefig(os.path.join(path, "weighted_loss_validation.pdf"), format='pdf')


def weighted_loss(y_true, y_pred, positive_weights, negative_weights):
    # clip to prevent NaN's and Inf's
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7, name=None)
    #y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = (-y_true * tf.log(y_pred) * positive_weights) - ((1.0 - y_true) * tf.log(1.0 - y_pred) * negative_weights)
    loss = tf.reduce_mean(loss)
    return loss


def positive_loss(y_true, y_pred, positive_weights):
    # clip to prevent NaN's and Inf's
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7, name=None)
    #y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = (-y_true * tf.log(y_pred) * positive_weights)
    loss = tf.reduce_mean(loss)
    return loss


def negative_loss(y_true, y_pred, negative_weights):
    # clip to prevent NaN's and Inf's
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7, name=None)
    #y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = -((1.0 - y_true) * tf.log(1.0 - y_pred) * negative_weights)
    loss = tf.reduce_mean(loss)
    return loss


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def main():
    print("Current Experiment: " + EXPERIMENTNAME + "\n\n\n")
    # Loading datasets
    training_dataset = get_training_dataset(os.path.join(SOURCE_PATH, "GroundTruth/train_ground_truth.csv"))
    val_dataset = get_validation_dataset(os.path.join(SOURCE_PATH, "GroundTruth/validation_ground_truth.csv"))

    # Setting up model
    y = tf.placeholder(tf.float32, [None, 15], name="true_labels")
    x_input = tf.placeholder(tf.float32, [None, 646, 96, 1], name="input")
    positive_weights = tf.placeholder(tf.float32, [None,15], name = "Positive_weights")
    negative_weights = tf.placeholder(tf.float32, [None, 15], name="negative_weights")
    current_keep_prob = tf.placeholder(tf.float32, name="dropout_rate")
    train_phase = tf.placeholder(tf.bool, name="is_training")
    logits, model_output = get_model(x_input, current_keep_prob, train_phase)

    # Defining loss and metrics
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
    my_weights_loss = weighted_loss(y_true= y, y_pred= model_output,
                                  positive_weights= positive_weights, negative_weights= negative_weights)
    my_positive_loss = positive_loss(y_true= y, y_pred= model_output, positive_weights= positive_weights)
    my_negative_loss = negative_loss(y_true= y, y_pred= model_output, negative_weights= negative_weights)

    # Learning rate decay
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.1, global_step=global_step, decay_steps=1000,
                                              decay_rate=0.95,staircase=True)
    '''
    These following lines are needed for batch normalization to work properly
    check https://timodenk.com/blog/tensorflow-batch-normalization/
    '''
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(my_weights_loss)
    correct_prediction = tf.equal(tf.round(model_output), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Adding tensorboard summaries
    tf.summary.scalar('Original cross_entropy', loss)
    tf.summary.scalar('Weighted cross entropy',  my_weights_loss)
    tf.summary.scalar('Accuracy', accuracy)
    tf.summary.scalar('Weighted positive loss',  my_positive_loss)
    tf.summary.scalar('Weighted negative loss',  my_negative_loss)
    # Merge all the summaries
    merged = tf.summary.merge_all()

    # Setting up dataset iterator
    training_iterator = training_dataset.make_one_shot_iterator()
    training_next_element = training_iterator.get_next()
    validation_iterator = val_dataset.make_one_shot_iterator()
    validation_next_element = validation_iterator.get_next()

    ## Setting up early stopping parameters
    # Best validation accuracy seen so far.
    best_validation_loss = 10e6  # Just some large number before storing the first validation loss
    # Iteration-number for last improvement to validation accuracy.
    last_improvement = 0
    # Stop optimization if no improvement found in this many iterations.
    min_epochs_for_early_stop = 10

    # Training paramaeters
    TRAINING_STEPS = 1053
    VALIDATION_STEPS = 156
    NUM_EPOCHS = 200

    # Setting up saving directory
    experiment_name = strftime("%Y-%m-%d_%H-%M-%S", localtime())
    exp_dir = os.path.join(OUTPUT_PATH, EXPERIMENTNAME, experiment_name)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    epoch_losses_history, epoch_accurcies_history, val_losses_history, val_accuracies_history = [], [], [], []
    my_loss_history, my_loss_val_history = [], []
    with tf.Session() as sess:
        # Write summaries to LOG_DIR -- used by TensorBoard
        train_writer = tf.summary.FileWriter(exp_dir + '/tensorboard/train', graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter(exp_dir + '/tensorboard/test', graph=tf.get_default_graph())
        print("Execute the following in a terminal:\n" + "tensorboard --logdir=" + exp_dir)
        sess.run(tf.global_variables_initializer())
        for epoch in range(NUM_EPOCHS):
            batch_loss, batch_accuracy = np.zeros([TRAINING_STEPS, 1]), np.zeros([TRAINING_STEPS, 1])
            batch_my_loss, val_my_loss = np.zeros([TRAINING_STEPS, 1]), np.zeros([VALIDATION_STEPS, 1])
            val_accuracies, val_losses = np.zeros([VALIDATION_STEPS, 1]), np.zeros([VALIDATION_STEPS, 1])
            for batch_counter in range(TRAINING_STEPS):
                batch = sess.run(training_next_element)
                summary, batch_loss[batch_counter], batch_accuracy[batch_counter], batch_my_loss[batch_counter],  _ = sess.run([merged, loss, accuracy, my_weights_loss, train_step],
                                                                                       feed_dict={x_input: batch[0],
                                                                                                  y: batch[1],
                                                                                                  positive_weights: batch[2],
                                                                                                  negative_weights: batch[3],
                                                                                                  current_keep_prob: 0.3,
                                                                                                  train_phase: True})
            print("Epoch #{}".format(epoch+1), "Loss: {:.4f}".format(np.mean(batch_loss)), "My_loss: {:.4f}".format(np.mean(batch_my_loss)),
                  "accuracy: {:.4f}".format(np.mean(batch_accuracy)))
            epoch_losses_history.append(np.mean(batch_loss)); epoch_accurcies_history.append(np.mean(batch_accuracy))
            my_loss_history.append(np.mean(batch_my_loss))
            # Add to summaries
            train_writer.add_summary(summary, epoch)


            for validation_batch in range(VALIDATION_STEPS):
                val_batch = sess.run(validation_next_element)
                summary, val_losses[validation_batch], val_accuracies[validation_batch], val_my_loss[validation_batch] = sess.run([merged, loss, accuracy, my_weights_loss],
                                                                                          feed_dict={
                                                                                              x_input: val_batch[0],
                                                                                              y: val_batch[1],
                                                                                              positive_weights: val_batch[2],
                                                                                              negative_weights: val_batch[3],
                                                                                              current_keep_prob: 1,
                                                                                              train_phase: False})
            print("validation Loss : {:.4f}".format(np.mean(val_losses)),
                  "validation accuracy: {:.4f}".format(np.mean(val_accuracies)))
            val_losses_history.append(np.mean(val_losses)); val_accuracies_history.append(np.mean(val_accuracies))
            my_loss_val_history.append(np.mean(val_my_loss))
            test_writer.add_summary(summary, epoch)


            # If validation loss is an improvement over best-known.
            if np.mean(val_my_loss) < best_validation_loss:
                # Update the best-known validation accuracy.
                best_validation_loss = np.mean(val_my_loss)

                # Set the iteration for the last improvement to current.
                last_improvement = epoch

                # Save all variables of the TensorFlow graph to file.
                save_path = saver.save(sess, os.path.join(exp_dir, "best_validation.ckpt"))
                #print("Model with best validation saved in path: %s" % save_path)

            # If no improvement found in the required number of iterations.
            if epoch - last_improvement > min_epochs_for_early_stop:
                print("No improvement found in a last 10 epochs, stopping optimization.")
                # Break out from the for-loop.
                break

        save_path = saver.save(sess, os.path.join(exp_dir, "last_epoch.ckpt"))
        print("Last iteration model saved in path: %s" % save_path)

        # Loading model with best validation
        saver.restore(sess, os.path.join(exp_dir, "best_validation.ckpt"))
        print("Model with best validation restored before testing.")
        # Testing the model [I split the testset into smaller splits because of memory error]
        spectrograms, test_classes = load_test_set_raw()
        TEST_NUM_STEPS = 283  # number is chosen based on testset size to be dividable [would change based on dataset]
        split_size = int(len(test_classes) / TEST_NUM_STEPS)
        test_pred_prob = np.zeros_like(test_classes, dtype=float)
        for test_split in range(TEST_NUM_STEPS):
            spectrograms_split = spectrograms[(test_split * split_size):(test_split * split_size) + split_size, :, :]
            test_pred_prob[(test_split * split_size):(test_split * split_size) + split_size, :] = sess.run(model_output,
                                                                                                           feed_dict={
                                                                                                               x_input: spectrograms_split,
                                                                                                               current_keep_prob: 1,
                                                                                                               train_phase: False})
        accuracy, auc_roc, hamming_error = evaluate_model(test_pred_prob, test_classes,
                                                          saving_path=exp_dir,
                                                          evaluation_file_path=os.path.join(exp_dir,
                                                                                            "evaluation_results.txt"))

        # Running on validation set to adjust the threshold values
        # Testing the model [I split the testset into smaller splits because of memory error]
        va_spectrograms, val_classes = load_validation_set_raw()
        VAL_NUM_STEPS = 607  # number is chosen based on testset size to be dividabable [would change based on dataset]
        split_size = int(len(val_classes) / VAL_NUM_STEPS)
        val_pred_prob = np.zeros_like(val_classes, dtype=float)
        for val_split in range(VAL_NUM_STEPS):
            spectrograms_split = va_spectrograms[(val_split * split_size):(val_split * split_size) + split_size, :, :]
            val_pred_prob[(val_split * split_size):(val_split * split_size) + split_size, :] = sess.run(model_output,
                                                                                                           feed_dict={
                                                                                                               x_input: spectrograms_split,
                                                                                                               current_keep_prob: 1,
                                                                                                               train_phase: False})
        results = create_analysis_report(test_pred_prob, test_classes, exp_dir, LABELS_LIST, val_pred_prob, val_classes)

    # Plot and save losses
    plot_loss_acuracy(epoch_losses_history, epoch_accurcies_history, val_losses_history, val_accuracies_history, exp_dir)
    plot_new_old_loss(epoch_losses_history, my_loss_history, exp_dir)
    plot_my_loss_validation(my_loss_history, my_loss_val_history, exp_dir)


if __name__ == "__main__":
    main()
