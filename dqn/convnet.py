import tensorflow as tf
from functools import reduce

# Creates the DQN
# Input format to the conv layers is the default "NHWC" : [batch, height, width, channels]


def create_network(x):
    initializer = tf.truncated_normal_initializer(0, 0.02)

    # DQN implementation
    with tf.variable_scope('dqn'):

        # First layer of DQN
        with tf.variable_scope('layer1'):
            stride1 = [1, 4, 4, 1]
            kernel_shape1 = [8, 8, x.get_shape()[-1], 32]

            # Convolution layer
            w1 = tf.get_variable('w1', kernel_shape1, tf.float32, initializer=initializer)
            conv1 = tf.nn.conv2d(x, w1, stride1, 'VALID', name='conv1')

            # Add the bias term
            b1 = tf.get_variable('b1', [32], initializer=tf.constant_initializer(0.0))
            z1 = tf.nn.bias_add(conv1, b1, data_format='NHWC', name='z1')

            # Nonlinearity
            out1 = tf.nn.relu(z1, name='out1')

        # Second layer of DQN
        with tf.variable_scope('layer2'):
            stride2 = [1, 2, 2, 1]
            kernel_shape2 = [4, 4, out1.get_shape()[-1], 64]

            # Convolution layer
            w2 = tf.get_variable('w2', kernel_shape2, tf.float32, initializer=initializer)
            conv2 = tf.nn.conv2d(out1, w2, stride2, 'VALID', name='conv2')

            # Add the bias term
            b2 = tf.get_variable('b2', [64], initializer=tf.constant_initializer(0.0))
            z2 = tf.nn.bias_add(conv2, b2, data_format='NHWC', name='z2')

            # Nonlinearity
            out2 = tf.nn.relu(z2, name='out2')

        # Third layer of DQN
        with tf.variable_scope('layer3'):
            stride3 = [1, 1, 1, 1]
            kernel_shape3 = [3, 3, out2.get_shape()[-1], 64]

            # Convolution layer
            w3 = tf.get_variable('w3', kernel_shape3, tf.float32, initializer=initializer)
            conv3 = tf.nn.conv2d(out2, w3, stride3, 'VALID', name='conv3')

            # Add the bias term
            b3 = tf.get_variable('b3', [64], initializer=tf.constant_initializer(0.0))
            z3 = tf.nn.bias_add(conv3, b3, data_format='NHWC', name='z3')

            # Nonlinearity
            out3 = tf.nn.relu(z3, name='out3')

        # Flatten the third layer
        out3_shape = out3.get_shape().as_list()
        out3_flat = tf.reshape(out3, [-1, reduce(lambda x, y: x * y, out3_shape[1:])])

        # Finaly hidden layer, fully connected
        with tf.variable_scope('layer4'):
            out3_flat_shape = out3_flat.get_shape().as_list()

            w4 = tf.get_variable('w4', [out3_flat_shape[1], 512], tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
            b4 = tf.get_variable('b4', [512], initializer=tf.constant_initializer(0.0))

            z4 = tf.nn.bias_add(tf.matmul(out3_flat, w4), b4, name='z4')

            # Nonlinearity
            out4 = tf.nn.relu(z4, name='out4')

        # Output layer
        with tf.variable_scope('output'):
            out4_shape = out4.get_shape().as_list()

            w5 = tf.get_variable('w5', [out4_shape[1], 4], tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
            b5 = tf.get_variable('b5', [4], initializer=tf.constant_initializer(0.0))

            z5 = tf.nn.bias_add(tf.matmul(out4, w5), b5, name='z5')

            # Nonlinearity
            q = tf.nn.relu(z5, name='q')

    return q
