# MUSA MAHMOOD - Copyright 2017
# Python 3.6.1
# TF 1.4.1

# IMPORTS:
import tensorflow as tf
import os.path as path
import pandas as pd
import numpy as np
import os as os
import datetime
import glob
import time

from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

# CONSTANTS:
TIMESTAMP_START = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H.%M.%S')
VERSION_NUMBER = 'v0.2.0'
win_len = 128
TRAINING_FOLDER_PATH = r'output_dir/hpf_scaled_up_4c_' + str(win_len)
DESCRIPTION_TRAINING_DATA = 'PSD_1ch_EMG'
TEST_FOLDER_PATH = TRAINING_FOLDER_PATH + '/v'
EXPORT_DIRECTORY = 'model_exports_' + VERSION_NUMBER + '/'
MODEL_NAME = 'emg_2cnn_1ch_wlen' + str(win_len)
CHECKPOINT_FILE = EXPORT_DIRECTORY + MODEL_NAME + '.ckpt'

# MATLAB DICT KEYS
KEY_X_DATA_DICTIONARY = 'relevant_data'
KEY_Y_DATA_DICTIONARY = 'Y'

# IMAGE SHAPE/CHARACTERISTIC1
DATA_WINDOW_SIZE = win_len
NUMBER_CLASSES = 4
SELECT_DATA_CHANNELS = np.asarray(range(1, 2))
NUMBER_DATA_CHANNELS = SELECT_DATA_CHANNELS.shape[0]  # Selects first int in shape
TOTAL_DATA_CHANNELS = NUMBER_DATA_CHANNELS
DEFAULT_IMAGE_SHAPE = [TOTAL_DATA_CHANNELS, DATA_WINDOW_SIZE]
INPUT_IMAGE_SHAPE = [1, TOTAL_DATA_CHANNELS, DATA_WINDOW_SIZE]

# FOR MODEL DESIGN
NUMBER_STEPS = 20000
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 100
LEARNING_RATE = 1e-5  # 'Step size' on n-D optimization plane
STRIDE_CONV2D_L1 = [1, 1, 1, 1]
STRIDE_CONV2D_L2 = [1, 1, 1, 1]

MAX_POOL_K_SIZE = [1, 1, 2, 1]
MAX_POOL_STRIDE = [1, 1, 2, 1]  # May need to reduce batch size for this.

BIAS_VAR_CL1 = 32
BIAS_VAR_CL2 = 64

DIVIDER = 4

WEIGHT_VAR_CL1 = [4, 4, 1, BIAS_VAR_CL1]  # [5, NUMBER_DATA_CHANNELS, 1, 32]
WEIGHT_VAR_CL2 = [4, 4, BIAS_VAR_CL1, BIAS_VAR_CL2]  # [5, NUMBER_DATA_CHANNELS, 32, 64]

MAX_POOL_FLAT_SHAPE_FC1 = [-1, NUMBER_DATA_CHANNELS * (DATA_WINDOW_SIZE // DIVIDER) * BIAS_VAR_CL2]

BIAS_VAR_FC1 = [1024]
WEIGHT_VAR_FC_OUTPUT = [*BIAS_VAR_FC1, NUMBER_CLASSES]
WEIGHT_VAR_FC1 = [(DATA_WINDOW_SIZE // DIVIDER) * NUMBER_DATA_CHANNELS * BIAS_VAR_CL2, *BIAS_VAR_FC1]

BIAS_VAR_FC_OUTPUT = [NUMBER_CLASSES]

# Start Script Here:
if not path.exists(EXPORT_DIRECTORY):
    os.mkdir(EXPORT_DIRECTORY)
input_node_name = 'input'
keep_prob_node_name = 'keep_prob'
output_node_name = 'output'


def load_data(data_directory):
    x_train_data = np.empty([0, *DEFAULT_IMAGE_SHAPE], np.float32)
    y_train_data = np.empty([0], np.float32)
    training_files = glob.glob(data_directory + "/*.mat")
    for f in training_files:
        x_array = loadmat(f).get(KEY_X_DATA_DICTIONARY)
        y_array = loadmat(f).get(KEY_Y_DATA_DICTIONARY)
        y_array = y_array.reshape([np.amax(y_array.shape)])
        x_train_data = np.concatenate((x_train_data, x_array), axis=0)
        y_train_data = np.concatenate((y_train_data, y_array), axis=0)
    y_train_data = np.asarray(pd.get_dummies(y_train_data).values).astype(np.float32)
    # return data_array
    print("Loaded Data Shape: X:", x_train_data.shape, " Y: ", y_train_data.shape)
    return x_train_data, y_train_data


def export_model(input_node_names, output_node_name_internal):
    freeze_graph.freeze_graph(EXPORT_DIRECTORY + MODEL_NAME + '.pbtxt', None, False,
                              EXPORT_DIRECTORY + MODEL_NAME + '.ckpt', output_node_name_internal, "save/restore_all",
                              "save/Const:0", EXPORT_DIRECTORY + '/frozen_' + MODEL_NAME + '.pb', True, "")
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(EXPORT_DIRECTORY + '/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, input_node_names, [output_node_name_internal], tf.float32.as_datatype_enum)
    with tf.gfile.FastGFile(EXPORT_DIRECTORY + '/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("Graph Saved - Output Directories: ")
    print("1 - Standard Frozen Model:", EXPORT_DIRECTORY + '/frozen_' + MODEL_NAME + '.pb')
    print("2 - Android Optimized Model:", EXPORT_DIRECTORY + '/opt_' + MODEL_NAME + '.pb')


# Model Building Macros: #
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def tf_conv(x_, w_, b_, strides):
    x_ = tf.nn.conv2d(x_, w_, strides=strides, padding='SAME')
    x_ = tf.nn.bias_add(x_, b_)
    return tf.nn.relu(x_)


def tf_max_pool(x_, kernel_size, strides):
    return tf.nn.max_pool(x_, ksize=kernel_size, strides=strides, padding='SAME')


def get_activations(layer, input_val, shape, directory, file_name, sum_all=False):
    os.makedirs(directory)
    units = sess.run(layer, feed_dict={x: np.reshape(input_val, shape, order='F'), keep_prob: 1.0})
    print("units.shape: ", units.shape)
    # plot_nn_filter(units, directory + file_name, True)
    new_shape = [units.shape[1], units.shape[2]]
    feature_maps = units.shape[3]
    filename_ = directory + file_name
    if sum_all:
        new_array = np.reshape(units.sum(axis=3), new_shape)
        pd.DataFrame(new_array).to_csv(filename_ + '_weight_matrix' + '.csv', index=False, header=False)
        major_axis = np.argmax(new_array.shape)
        summed_array = new_array.sum(axis=major_axis)
        pd.DataFrame(summed_array).to_csv(filename_ + '_sum_all' + '.csv', index=False, header=False)
        print('All Values:')
        return summed_array
    else:
        for i0 in range(feature_maps):
            pd.DataFrame(units[:, :, :, i0].reshape(new_shape)).to_csv(
                filename_ + '_' + str(i0 + 1) + '.csv', index=False, header=False)
        return units


def get_activations_mat(layer, input_val, shape):
    units = sess.run(layer, feed_dict={x: np.reshape(input_val, shape, order='F'), keep_prob: 1.0})
    # print("units.shape: ", units.shape)
    return units


def get_all_activations(training_data, folder_name):
    w_hconv1 = np.empty([0, *h_conv1_shape[1:]], np.float32)
    w_hpool1 = np.empty([0, *h_pool1_shape[1:]], np.float32)
    w_hconv2 = np.empty([0, *h_conv2_shape[1:]], np.float32)
    w_hpool2 = np.empty([0, *h_pool2_shape[1:]], np.float32)
    w_hpool2_flat = np.empty([0, h_pool2_flat_shape[1]], np.float32)
    w_hfc1 = np.empty([0, h_fc1_shape[1]], np.float32)
    w_hfc1_do = np.empty([0, h_fc1_drop_shape[1]], np.float32)
    w_y_out = np.empty([0, y_conv_shape[1]], np.float32)
    print('Getting all Activations: please wait... ')
    for it in range(0, training_data.shape[0]):
        if it % 100 == 0:
            print('Saved Sample #', it)
        sample = training_data[it]
        w_hconv1 = np.concatenate((w_hconv1, get_activations_mat(h_conv1, sample, INPUT_IMAGE_SHAPE)), axis=0)
        w_hpool1 = np.concatenate((w_hpool1, get_activations_mat(h_pool1, sample, INPUT_IMAGE_SHAPE)), axis=0)
        w_hconv2 = np.concatenate((w_hconv2, get_activations_mat(h_conv2, sample, INPUT_IMAGE_SHAPE)), axis=0)
        w_hpool2 = np.concatenate((w_hpool2, get_activations_mat(h_pool2, sample, INPUT_IMAGE_SHAPE)), axis=0)
        w_hpool2_flat = np.concatenate((w_hpool2_flat, get_activations_mat(h_pool2_flat, sample, INPUT_IMAGE_SHAPE)),
                                       axis=0)
        w_hfc1 = np.concatenate((w_hfc1, get_activations_mat(h_fc1, sample, INPUT_IMAGE_SHAPE)), axis=0)
        w_hfc1_do = np.concatenate((w_hfc1_do, get_activations_mat(h_fc1_drop, sample, INPUT_IMAGE_SHAPE)), axis=0)
        w_y_out = np.concatenate((w_y_out, get_activations_mat(y_conv, sample, INPUT_IMAGE_SHAPE)), axis=0)
        # Save all activations:
    fn_out = folder_name + 'all_activations.mat'
    savemat(fn_out, mdict={'input_sample': training_data, 'h_conv1': w_hconv1, 'h_pool1': w_hpool1, 'h_conv2': w_hconv2,
                           'h_pool2': w_hpool2,
                           'h_pool2_flat': w_hpool2_flat, 'h_fc1': w_hfc1, 'h_fc1_drop': w_hfc1_do,
                           'y_out': w_y_out})


# MODEL INPUT #
x = tf.placeholder(tf.float32, shape=[None, *DEFAULT_IMAGE_SHAPE], name=input_node_name)
keep_prob = tf.placeholder(tf.float32, name=keep_prob_node_name)
y = tf.placeholder(tf.float32, shape=[None, NUMBER_CLASSES])

x_input = tf.reshape(x, [-1, *DEFAULT_IMAGE_SHAPE, 1])

# first convolution and pooling
W_conv1 = weight_variable(WEIGHT_VAR_CL1)
b_conv1 = bias_variable([BIAS_VAR_CL1])

# h_conv1 = tf.nn.relu(conv2d(x_input, W_conv1) + b_conv1)
h_conv1 = tf_conv(x_input, W_conv1, b_conv1, STRIDE_CONV2D_L1)
h_pool1 = tf_max_pool(h_conv1, MAX_POOL_K_SIZE, MAX_POOL_STRIDE)

# second convolution and pooling
W_conv2 = weight_variable(WEIGHT_VAR_CL2)
b_conv2 = bias_variable([BIAS_VAR_CL2])

# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_conv2 = tf_conv(h_pool1, W_conv2, b_conv2, STRIDE_CONV2D_L2)
h_pool2 = tf_max_pool(h_conv2, MAX_POOL_K_SIZE, MAX_POOL_STRIDE)

# the input should be shaped/flattened
h_pool2_flat = tf.reshape(h_pool2, MAX_POOL_FLAT_SHAPE_FC1)

# fully connected layer1,the shape of the patch should be defined
W_fc1 = weight_variable(WEIGHT_VAR_FC1)
b_fc1 = bias_variable(BIAS_VAR_FC1)

h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool2_flat, W_fc1), b_fc1))

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# weight and bias of the output layer
W_fco = weight_variable(WEIGHT_VAR_FC_OUTPUT)
b_fco = bias_variable(BIAS_VAR_FC_OUTPUT)

y_conv = tf.add(tf.matmul(h_fc1_drop, W_fco), b_fco)

outputs = tf.nn.softmax(y_conv, name=output_node_name)

prediction = tf.argmax(outputs, 1)

# training and reducing the cost/loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver()  # Initialize tf Saver

# Load Data:
x_data, y_data = load_data(TRAINING_FOLDER_PATH)
x_val_data, y_val_data = load_data(TEST_FOLDER_PATH)
# Split training set:
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.75, random_state=1)

# TRAIN ROUTINE #
init_op = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

val_step = 0
with tf.Session(config=config) as sess:
    sess.run(init_op)

    x_0 = np.zeros(INPUT_IMAGE_SHAPE, dtype=np.float32)
    print("Model Dimensions: ")
    h_conv1_shape = sess.run(h_conv1, feed_dict={x: x_0, keep_prob: 1.0}).shape
    h_pool1_shape = sess.run(h_pool1, feed_dict={x: x_0, keep_prob: 1.0}).shape
    h_conv2_shape = sess.run(h_conv2, feed_dict={x: x_0, keep_prob: 1.0}).shape
    h_pool2_shape = sess.run(h_pool2, feed_dict={x: x_0, keep_prob: 1.0}).shape
    h_pool2_flat_shape = sess.run(h_pool2_flat, feed_dict={x: x_0, keep_prob: 1.0}).shape
    h_fc1_shape = sess.run(h_fc1, feed_dict={x: x_0, keep_prob: 1.0}).shape
    h_fc1_drop_shape = sess.run(h_fc1_drop, feed_dict={x: x_0, keep_prob: 1.0}).shape
    y_conv_shape = sess.run(y_conv, feed_dict={x: x_0, keep_prob: 1.0}).shape
    print("h_conv1: ", h_conv1_shape)
    print("h_pool1: ", h_pool1_shape)
    print("h_conv2: ", h_conv2_shape)
    print("h_pool2: ", h_pool2_shape)
    print("h_pool2_flat: ", h_pool2_flat_shape)
    print("h_fc1: ", h_fc1_shape)
    print("h_fc1_drop: ", h_fc1_drop_shape)
    print("y_conv: ", y_conv_shape)

    # save model as pbtxt:
    tf.train.write_graph(sess.graph_def, EXPORT_DIRECTORY, MODEL_NAME + '.pbtxt', True)

    for i in range(NUMBER_STEPS):
        offset = (i * TRAIN_BATCH_SIZE) % (x_train.shape[0] - TRAIN_BATCH_SIZE)
        batch_x_train = x_train[offset:(offset + TRAIN_BATCH_SIZE)]
        batch_y_train = y_train[offset:(offset + TRAIN_BATCH_SIZE)]
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_x_train, y: batch_y_train, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

        if i % 20 == 0:
            # Calculate batch loss and accuracy
            offset = (val_step * TEST_BATCH_SIZE) % (x_test.shape[0] - TEST_BATCH_SIZE)
            batch_x_val = x_test[offset:(offset + TEST_BATCH_SIZE), :, :]
            batch_y_val = y_test[offset:(offset + TEST_BATCH_SIZE), :]
            val_accuracy = accuracy.eval(feed_dict={x: batch_x_val, y: batch_y_val, keep_prob: 1.0})
            print("Validation step %d, validation accuracy %g" % (val_step, val_accuracy))
            val_step += 1

        train_step.run(feed_dict={x: batch_x_train, y: batch_y_train, keep_prob: 0.25})

    # Run test data (entire set) to see accuracy.
    test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y: y_test, keep_prob: 1.0})  # original
    print("\n Testing Accuracy:", test_accuracy, "\n\n")

    # Holdout Validation Accuracy:
    print("Holdout Validation:", sess.run(accuracy, feed_dict={x: x_val_data, y: y_val_data,
                                                               keep_prob: 1.0}))
    y_val_tf = np.zeros([x_val_data.shape[0]], dtype=np.int32)
    predictions = np.zeros([x_val_data.shape[0]], dtype=np.int32)
    for i in range(0, x_val_data.shape[0]):
        predictions[i] = sess.run(prediction,
                                  feed_dict={x: x_val_data[i].reshape([1, NUMBER_DATA_CHANNELS, DATA_WINDOW_SIZE]),
                                             y: y_val_data[i].reshape([1, NUMBER_CLASSES]), keep_prob: 1.0})
        for c in range(0, NUMBER_CLASSES):
            if y_val_data[i][c]:
                y_val_tf[i] = c

    tf_confusion_matrix = tf.confusion_matrix(labels=y_val_tf, predictions=predictions, num_classes=NUMBER_CLASSES)
    print('Confusion Matrix: \n\n', tf.Tensor.eval(tf_confusion_matrix, feed_dict=None, session=None))

    user_input = input('Export Current Model?')
    if user_input == "1" or user_input.lower() == "y":
        saver.save(sess, CHECKPOINT_FILE)
        export_model([input_node_name, keep_prob_node_name], output_node_name)
