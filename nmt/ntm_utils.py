import numpy as np
import tensorflow as tf

def expand(x, dim, N, dims):
    return tf.tile(tf.expand_dims(x, dim), multiples=[N] + [1 for _ in range(dims)])

    # return tf.concat([tf.expand_dims(x, dim) for _ in tf.range(N)], axis=dim)

def learned_init(units):
    return tf.squeeze(tf.contrib.layers.fully_connected(tf.ones([1, 1]), units,
        activation_fn=None, biases_initializer=None))

def create_linear_initializer(input_size, dtype=tf.float32):
    stddev = 1.0 / np.sqrt(input_size)
    return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)