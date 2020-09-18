import tensorflow as tf
from tensorpack.models import *

def coords_grid(batch, ht, wd):
    with tf.name_scope('generate_grid'):
        coords = tf.meshgrid(tf.range(wd), tf.range(ht)) # list:[x,y]
        coords = tf.cast(tf.stack(coords, axis=-1), tf.float32) # [h,w,2]

        coords = tf.expand_dims(coords, 0) # [1,h,w,2]
        coords = tf.repeat(coords, repeats=batch, axis=0)
        return coords # [b,h,w,2]

def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)

def tf_grid_sample(img, coords):
    # coords: [b, h, w, 2] pixel coordinates
    _, H, W, _ = tf.unstack(tf.shape(img))


    flows = coords

    max_y = tf.cast(H - 1, tf.int32)
    max_x = tf.cast(W - 1, tf.int32)
    zero = tf.zeros([], dtype=tf.int32)

    x = flows[:, :, :, 0]
    y = flows[:, :, :, 1]
    x0 = x
    y0 = y
    x0 = tf.cast(x0, tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(y0, tf.int32)
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
    return out

def bilinear_sampler(img, coords):
    with tf.name_scope('grid_sample'):
        return tf_grid_sample(img, coords)

def upflow8(flow):
    with tf.name_scope('upflow8'):
        inp_shape = flow.shape.as_list()
        out_shape = tf.constant([inp_shape[1]*8, inp_shape[2]*8], tf.int32)

        up = tf.image.resize_bilinear(flow, out_shape, align_corners=True, name='output') # TODO: check half_pixel_center
        return up