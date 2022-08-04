from hamcrest import none
import tensorflow as tf
import tf_slim as slim
# from tensorflow.keras.utils import to_categorical
import numpy as np

# 计算成本,越小越好
def compute_cost(y_truth, y_pred):
    # y_truth = tf.reshape(y_truth,[27455, 25])
    # print(y_truth.shape, y_pred.shape)
    return tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_truth))


# placeholder
def create_placeholder(minibatch_size, n_H0, n_W0, n_C0, n_y):
    tf.compat.v1.disable_eager_execution()
    X = tf.compat.v1.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name = "X")
    Y = tf.compat.v1.placeholder(tf.float32, [None, n_y], name = "Y")
    
    return X, Y

import tensorflow as tf
# 初始化过滤器
def initialize_parameters():
    # [n_H, n_W, n_C, num] 分别是高 宽 通道数 输出通道数（过滤器数量）
    W1 = tf.Variable(tf.random.normal([3,3,1,4]))
    W2 = tf.Variable(tf.random.normal([3,3,4,12]))
    W3 = tf.Variable(tf.random.normal([3,3,12,16]))
    W4 = tf.Variable(tf.random.normal([3,3,16,12]))
    W5 = tf.Variable(tf.random.normal([3,3,12,4]))
    parameters = {
        "W1": W1,
        "W2": W2,
        "W3": W3,
        "W4": W4,
        "W5": W5
    }
    return parameters

# CNN前向传播
def build():
    # W1 = parameters["W1"]
    # W2 = parameters["W2"]
    # W3 = parameters["W3"]
    # W4 = parameters["W4"]
    # W5 = parameters["W5"]
    # X = tf.convert_to_tensor(X)
    inputs = tf.keras.Input(shape = (28,28,1))
    x = tf.keras.layers.Conv2D(filters = 4, kernel_size = 3, strides = 1, activation = "relu")(inputs)
    x = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = None, padding = 'same')(x)

    x = tf.keras.layers.Conv2D(filters = 4, kernel_size = 3, strides = 1, activation = "relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = None, padding = 'same')(x)
    # # conv1
    # Z1 = tf.nn.conv2d(inputs, W1, strides=[1,1,1,1], padding="SAME")
    # A1 = tf.nn.relu(Z1)
    # P1 = tf.nn.max_pool(A1, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME")
    # # conv2
    # Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding="SAME")
    # A2 = tf.nn.relu(Z2)
    # P2 = tf.nn.max_pool(A2, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME")
    # # conv3
    # Z3 = tf.nn.conv2d(P2, W3, strides=[1,1,1,1], padding="SAME")
    # A3 = tf.nn.relu(Z3)
    # # conv4
    # Z4 = tf.nn.conv2d(A3, W4, strides=[1,1,1,1], padding="SAME")
    # A4 = tf.nn.relu(Z4)
    # # conv5
    # Z5 = tf.nn.conv2d(A4, W5, strides=[1,1,1,1], padding="SAME")
    # A5 = tf.nn.relu(Z5)
    # P3 = tf.nn.max_pool(A5, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME")
    # FC
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(.1)(x)
    x = tf.keras.layers.Dense(128, activation = 'relu')(x)
    x = tf.keras.layers.Dropout(.1)(x)
    x = tf.keras.layers.Dense(25, activation = None)(x)
    # P = slim.flatten(P2)
    # P = slim.dropout(P,0.9)
    # Z6 = slim.fully_connected(P, 128, activation_fn=tf.nn.relu)
    # Z6 = slim.dropout(Z6,0.9)
    # Z7 = slim.fully_connected(Z6, 25, activation_fn=None)
    # print("Output size:", x.shape)
    model = tf.keras.Model(inputs = inputs, outputs = x)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.01), loss = compute_cost, metrics = "accuracy")
    # print("saving")
    # Z7 = slim.dropout(Z7,0.9)
    # Z8 = slim.fully_connected(Z7,25,activation_fn=None)
    return model


    
    
