# General Imports
import os
import numpy as np
import pandas as pd
from time import strftime, localtime
import matplotlib.pyplot as plt
from utilities import create_analysis_report

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


EXPERIMENTNAME = "C4_square_tf"
INPUT_SHAPE = (646, 96, 1)
LABELS_LIST = ['car', 'chill', 'club', 'dance', 'gym', 'happy', 'night', 'party', 'relax', 'running',
               'sad', 'sleep', 'summer', 'work', 'workout']


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

    dataset = dataset.map(lambda sample: dict(sample, binary_label=tf_multilabel_binarize(
        sample.get("label", b""), label_list_tf=tf.constant(LABELS_LIST))[0]), )

    # set output shape
    dataset = dataset.map(lambda sample: dict(sample, binary_label=dp.set_tensor_shape(
        sample["binary_label"], (len(LABELS_LIST)))))


    """
    TEMPORARY TILL SONG_ID PASSING IS REPLACED BY LABEL PASSING [FOR THE CUSTOM LOSS FUNCTION]
    """
    #dataset = dataset.map(lambda sample: dict(sample, binary_label=tf_replace_labels_with_ID(
    #    sample.get("song_id"), label_list_tf=tf.constant(LABELS_LIST))[0]), )
    # set output shape
    #dataset = dataset.map(lambda sample: dict(sample, binary_label=dp.set_tensor_shape(
    #    sample["binary_label"], (len(LABELS_LIST)))))

    if infinite_generator:
        # Repeat indefinitly
        dataset = dataset.repeat(count=-1)

    # Make batch
    dataset = dataset.batch(batch_size)

    # Select only features and annotation
    dataset = dataset.map(lambda sample: (sample["features"], sample["binary_label"]))

    return dataset

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
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv_2d(x,W,name=""):
    return tf.nn.conv2d(x,W,[1,1,1,1],padding="SAME", name = name)

def max_pooling(x, shape, name = ""):
    return tf.nn.max_pool(x,shape,strides=[1,2,2,1],padding="SAME", name = name)

def conv_layer_with_reul(input, shape, name =""):
    W = get_weights(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv_2d(input, W, name)+b)

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = get_weights([in_size,size])
    b = bias_variable([size])
    return tf.matmul(input,W)+ b

def get_model(x_input,current_keep_prob):
    # Define model architecture
    # C4_model
    x_norm = tf.layers.batch_normalization(x_input, training=True)

    conv1 = conv_layer_with_reul(x_norm, [3, 3, 1, 32], name="conv_1")
    max1 = max_pooling(conv1, shape=[1, 2, 2, 1], name="max_pool_1")

    conv2 = conv_layer_with_reul(max1, [3, 3, 32, 64], name="conv_2")
    max2 = max_pooling(conv2, shape=[1, 2, 2, 1], name="max_pool_2")

    conv3 = conv_layer_with_reul(max2, [3, 3, 64, 128], name="conv_3")
    max3 = max_pooling(conv3, shape=[1, 2, 2, 1], name="max_pool_3")

    conv4 = conv_layer_with_reul(max3, [3, 3, 128, 256], name="conv_4")
    max4 = max_pooling(conv4, shape=[1, 2, 2, 1], name="max_pool_4")

    flattened = tf.reshape(max4, [-1, 41 * 6 * 256])
    fully1 = tf.nn.sigmoid(full_layer(flattened, 256))

    dropped = tf.nn.dropout(fully1, keep_prob=current_keep_prob)
    logits = full_layer(dropped, 15)
    output = tf.nn.sigmoid(logits)
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

    create_analysis_report(test_pred, test_classes, saving_path, LABELS_LIST)

    return accuracy, auc_roc, hamming_error


def main():
    # Loading datasets
    training_dataset = get_training_dataset(os.path.join(SOURCE_PATH, "GroundTruth/train_ground_truth.csv"))
    val_dataset = get_validation_dataset(os.path.join(SOURCE_PATH, "GroundTruth/validation_ground_truth.csv"))

    # Setting up model
    y = tf.placeholder(tf.float32, [None, 15], name = "true_labels")
    x_input = tf.placeholder(tf.float32, [None,646,96,1], name="input")
    current_keep_prob = tf.placeholder(tf.float32, name="dropout_rate")
    logits, model_output = get_model(x_input, current_keep_prob)

    # Defining loss and metrics
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
    '''
    These following lines are needed for batch normalization to work properly
    check https://timodenk.com/blog/tensorflow-batch-normalization/
    '''
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdadeltaOptimizer(learning_rate=0.01).minimize(loss)
    correct_prediction = tf.equal(tf.round(model_output) , y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Setting up dataset iterator
    training_iterator = training_dataset.make_one_shot_iterator()
    training_next_element = training_iterator.get_next()
    validation_iterator = val_dataset.make_one_shot_iterator()
    validation_next_element = validation_iterator.get_next()

    # Training paramaeters
    TRAINING_STEPS = 1053
    VALIDATION_STEPS = 156
    NUM_EPOCHS = 100

    # Setting up saving directory
    experiment_name = strftime("%Y-%m-%d_%H-%M-%S", localtime())
    exp_dir = os.path.join(OUTPUT_PATH, EXPERIMENTNAME,experiment_name)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(logdir=exp_dir, graph= sess.graph)
        for epoch in range(NUM_EPOCHS):
            batch_loss, batch_accuracy = np.zeros([TRAINING_STEPS, 1]), np.zeros([TRAINING_STEPS, 1])
            val_accuracies, val_losses = np.zeros([VALIDATION_STEPS, 1]), np.zeros([VALIDATION_STEPS, 1])
            for batch_counter in range (TRAINING_STEPS):
                if (batch_counter % 100 == 0):
                    print("batch # {}".format(batch_counter), " of Epoch # {}".format(epoch+1))
                batch = sess.run(training_next_element)
                batch_loss[batch_counter], batch_accuracy[batch_counter],_ = sess.run([loss, accuracy, train_step],
                                                                                      feed_dict={x_input: batch[0],
                                                                                        y: batch[1], current_keep_prob: 0.3})
                #print("Loss: {}".format(batch_loss), "accuracy: {}".format(batch_accuracy))
            print("Loss: {:.4f}".format(np.mean(batch_loss)), "accuracy: {:.4f}".format(np.mean(batch_accuracy)))

            for validation_batch in range(VALIDATION_STEPS):
                val_batch = sess.run(validation_next_element)
                val_losses[validation_batch], val_accuracies[validation_batch] = sess.run([loss, accuracy],
                                                                                          feed_dict={x_input: val_batch[0],
                                                                                            y: val_batch[1], current_keep_prob: 1})
            print("validation Loss : {:.4f}".format(np.mean(val_losses)) , "validation accuracy: {:.4f}".format(np.mean(val_accuracies)))

        save_path = saver.save(sess, os.path.join(exp_dir,"model.ckpt"))
        print("Model saved in path: %s" % save_path)

        # Testing the model [I split the testset into smaller splits because of memory error]
        spectrograms, test_classes = load_test_set_raw()
        TEST_NUM_STEPS = 100
        split_size = len(test_classes)/TEST_NUM_STEPS
        test_pred_prob = np.zeros_like(test_classes)
        for test_split in range(TEST_NUM_STEPS):
            spectrograms_split = spectrograms[test_split:split_size, :, :]
            test_classes_split = test_classes[test_split:split_size, :, :]
            test_pred_prob[test_split:split_size, :] = sess.run(model_output,
                                      feed_dict={x_input: spectrograms_split, y: test_classes_split, current_keep_prob: 1})
        #test_pred_prob = sess.run(model_output, feed_dict={x_input: spectrograms, y: test_classes, current_keep_prob: 1})
        accuracy, auc_roc, hamming_error = evaluate_model(test_pred_prob, test_classes,
                                                          saving_path=exp_dir,
                                                          evaluation_file_path=os.path.join(exp_dir, "evaluation_results.txt"))
    '''
    fit_config = {

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

    #compile_model(model, loss = loss,  optimizer=optimization, metrics=[originalCrossEntropymetric,positive_weighted_loss,negative_weighted_loss])

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
    '''

    """
    Ensure iterating through the whole dataset
                while True:
                try:
                    if (batch_counter % 20 == 0):
                        print("batch # {}".format(batch_counter), " of Epoch # {}".format(epoch+1))
                    batch = sess.run(training_next_element)
                    batch_loss,batch_accuracy,_ = sess.run([loss,accuracy, train_step], feed_dict={x_input: batch[0], y: batch[1], current_keep_prob: 0.3})
                    #print("Loss: {}".format(batch_loss), "accuracy: {}".format(batch_accuracy))
                    batch_counter += 1
                except tf.errors.OutOfRangeError:
                    break
    """
if __name__ == "__main__":
    main()
