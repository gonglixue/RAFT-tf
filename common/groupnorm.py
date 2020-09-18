import tensorflow as tf
from tensorpack.models import layer_register, BatchNorm

@layer_register()
def _groupnorm(x, group=32, epsilon=1e-5, training=None,
               beta_initializer=tf.constant_initializer(),
               gamma_initializer=tf.constant_initializer(1.)):
    """
    https://arxiv.org/abs/1803.08494
    TODO: Add a NHWC version.
    """
    # with tf.variable_scope(name):
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims == 4, shape
    chan = shape[1]
    # assert chan % group == 0, chan
    # dirty hack, please dont do this at home
    if chan == 728:
        group = 26

    group_size = chan // group

    orig_shape = tf.shape(x)
    h, w = orig_shape[2], orig_shape[3]

    x = tf.reshape(x, tf.stack([-1, group, group_size, h, w]))

    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)

    new_shape = [1, group, group_size, 1, 1]

    beta = tf.get_variable('beta', [chan], initializer=beta_initializer, trainable=training)
    beta = tf.reshape(beta, new_shape)

    gamma = tf.get_variable('gamma', [chan], initializer=gamma_initializer, trainable=training)
    gamma = tf.reshape(gamma, new_shape)

    out = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon, name='output')
    return tf.reshape(out, orig_shape, name='output')


def GroupNorm(x, group, name=None):
    return _groupnorm('gn', x, group=group)