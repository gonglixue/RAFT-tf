import tensorflow as tf
from tensorpack.models import *
from tensorpack.tfutils.tower import get_current_tower_context
from common.groupnorm import GroupNorm

def norm_func(inputs, name, norm_fn='group'):
    if norm_fn == 'group':
        num_group = inputs.get_shape().as_list()[3] // 8
        norm_out = GroupNorm(inputs, group=num_group)
    elif norm_fn == 'batch':
        norm_out = BatchNorm(name, inputs)
    elif norm_fn == 'instance':
        norm_out = InstanceNorm(name, inputs, center=False, scale=False)
    elif norm_fn == 'none':
        return inputs

    return norm_out

def ResidualBlock(inputs, out_planes, name='ResBlock', norm_fn='group', stride=1):
    res = inputs
    with tf.variable_scope(name):
        conv1_out = Conv2D('conv1', inputs, out_planes, kernel_size=3, strides=stride, padding='same')
        conv1_out = norm_func(conv1_out, name='norm1', norm_fn=norm_fn)
        conv1_out = tf.nn.relu(conv1_out)

        conv2_out = Conv2D('conv2', conv1_out, out_planes, kernel_size=3, padding='same')
        conv2_out = norm_func(conv2_out, name='norm2', norm_fn=norm_fn)
        y = tf.nn.relu(conv2_out)

        if stride == 1:
            return tf.nn.relu(res + y, name='output')
        else:
            res = Conv2D('downsample.0', res, out_planes, kernel_size=1, strides=stride)
            res = norm_func(res, name='downsample.1', norm_fn=norm_fn)
            return tf.nn.relu(res + y, name='output')

def BottleneckBlock(inputs, out_planes, name='Bottleneck', norm_fn='group', stride=1):
    res = inputs
    with tf.variable_scope(name):
        conv1_out = Conv2D('conv1', inputs, out_planes//4, kernel_size=1)
        conv1_out = norm_func(conv1_out, name='norm1', norm_fn=norm_fn)
        conv1_out = tf.nn.relu(conv1_out)

        conv2_out = Conv2D('conv2', conv1_out, out_planes//4, kernel_size=3, strides=stride)
        conv2_out = norm_func(conv2_out, name='norm2', norm_fn=norm_fn)
        conv2_out = tf.nn.relu(conv2_out)

        conv3_out = Conv2D('conv3', conv2_out, out_planes, kernel_size=1)
        conv3_out = norm_func(conv3_out, name='norm3', norm_fn=norm_fn)
        y = tf.nn.relu(conv3_out)

        if stride == 1:
            return tf.nn.relu(res + y, name='output')
        else:
            res = Conv2D('downsample.0', res, out_planes, kernel_size=1, strides=stride)
            res = norm_func(res, name='downsample.1', norm_fn=norm_fn)
            return tf.nn.relu(res + y, name='output')



def BasicEncoder(inputs, name, output_dim=256, norm_fn='instance', dropout=0.0):
    def _make_layer(inp, dim, name_, stride=1):
        with tf.variable_scope(name_):
            layer1_out = ResidualBlock(inp, dim, name='0', norm_fn=norm_fn, stride=stride)
            layer2_out = ResidualBlock(layer1_out, dim, name='1', norm_fn=norm_fn, stride=1)
            return layer2_out


    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv1_out = Conv2D('conv1', inputs, 64, kernel_size=7, strides=2) # os=2
        conv1_out = norm_func(conv1_out, name='norm1', norm_fn=norm_fn)
        conv1_out = tf.nn.relu(conv1_out)

        x = _make_layer(conv1_out, 64, name_='layer1', stride=1)    # os=2
        x = _make_layer(x, 96, name_='layer2', stride=2)            # os=4
        x = _make_layer(x, 128, name_='layer3', stride=2)           # os=8

        x = Conv2D('conv2', x, output_dim, kernel_size=1)

        if get_current_tower_context().is_training and dropout > 0: # TODO: no need to check if in training mode? Dropout will auto choose eval mode according to context
            x = Dropout(x, keep_prob=dropout) # TODO: In pytorch, the arg is drop_prob rather than keep_prob

        return x

def SmallEncoder(inputs, name, out_dim, norm_fn='batch', dropout=0.0):
    def _make_layer(inp, dim, name_, stride=1):
        with tf.variable_scope(name_):
            layer1_out = BottleneckBlock(inp, dim, name='0', norm_fn=norm_fn, stride=stride)
            layer2_out = BottleneckBlock(layer1_out, dim, name='1', norm_fn=norm_fn, stride=1)
            return layer2_out

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv1_out = Conv2D('conv1', inputs, 32, kernel_size=7, strides=2)   # os=2
        conv1_out = norm_func(conv1_out, name='norm1', norm_fn=norm_fn)
        conv1_out = tf.nn.relu(conv1_out)

        x = _make_layer(conv1_out, 32, name_='layer1', stride=1)
        x = _make_layer(x, 64, name_='layer2', stride=2)
        x = _make_layer(x, 96, name_='layer3', stride=2)

        x = Conv2D('conv2', x, out_dim, kernel_size=1)

        if get_current_tower_context().is_training and dropout > 0: # TODO: no need to check if in training mode. Dropout will auto choose eval mode according to context
            x = Dropout(x, keep_prob=dropout) # TODO: In pytorch, the arg is drop_prob rather than keep_prob
        return x



######################## update module #################################
def BasicMotionEncoder(flow, corr, name):
    with tf.variable_scope(name):
        cor = Conv2D('convc1', corr, 256, kernel_size=1, activation=tf.nn.relu)
        cor = Conv2D('convc2', cor, 192, kernel_size=3, activation=tf.nn.relu)
        flo = Conv2D('convf1', flow, 128, kernel_size=7, activation=tf.nn.relu)
        flo = Conv2D('convf2', flo, 64, kernel_size=3, activation=tf.nn.relu)

        cor_flo = tf.concat([cor, flo], axis=-1)
        out = Conv2D('conv', cor_flo, 128-2, kernel_size=3, activation=tf.nn.relu)
        return tf.concat([out, flow], axis=-1, name='concat_out')

def SmallMotionEncoder(flow, corr, name):
    with tf.variable_scope(name):
        cor = Conv2D('convc1', corr, 96, kernel_size=1, activation=tf.nn.relu)
        flo = Conv2D('convf1', flow, 64, kernel_size=7, activation=tf.nn.relu)
        flo = Conv2D('convf2', flo, 32, kernel_size=3, activation=tf.nn.relu)

        cor_flo = tf.concat([cor, flo], axis=-1)
        out = Conv2D('conv', cor_flo, 80, kernel_size=3, activation=tf.nn.relu)
        return tf.concat([out, flow], axis=-1, name='concat_out')

def FlowHead(x, name, hidden_dim=128):
    with tf.variable_scope(name):
        conv1 = Conv2D('conv1', x, hidden_dim, kernel_size=3, activation=tf.nn.relu)
        conv2 = Conv2D('conv2', conv1, 2, kernel_size=3)
        return conv2


def SepConvGRU(h, x, name, hidden_dim=128):
    with tf.variable_scope(name):
        # horizontal
        hx = tf.concat([h, x], axis=-1, name='cat_hx_horizontal')
        z = Conv2D('convz1', hx, hidden_dim, kernel_size=(1,5), activation=tf.sigmoid)
        r = Conv2D('convr1', hx, hidden_dim, kernel_size=(1,5), activation=tf.sigmoid)
        q_input = tf.concat([r*h, x], axis=-1, name='q_input_cat')
        q = Conv2D('convq1', q_input, hidden_dim, kernel_size=(1, 5), activation=tf.tanh)
        h = (1-z) * h + z*q

        # vertical
        hx = tf.concat([h, x], axis=-1, name='cat_hx_vertical')
        z = Conv2D('convz2', hx, hidden_dim, kernel_size=(5, 1), activation=tf.sigmoid)
        r = Conv2D('convr2', hx, hidden_dim, kernel_size=(5, 1), activation=tf.sigmoid)
        q_input = tf.concat([r*h, x], axis=-1, name='q_input_cat2')
        q = Conv2D('convq2', q_input, hidden_dim, kernel_size=(5, 1), activation=tf.tanh)
        h = (1-z) * h + z*q

        return h

def ConvGRU(h, x, name, hidden_dim=128):
    with tf.variable_scope(name):
        hx = tf.concat([h, x], axis=-1, name='cat_hx')

        z = Conv2D('convz', hx, hidden_dim, kernel_size=3, activation=tf.sigmoid)
        r = Conv2D('convr', hx, hidden_dim, kernel_size=3, activation=tf.sigmoid)

        q_input = tf.concat([r*h, x], axis=-1, name='q_input_cat')
        q = Conv2D('convq', q_input, hidden_dim, kernel_size=3, activation=tf.tanh)

        h = (1-z) * h + z * q
        return h


def BasicUpdateBlock(net, inp, corr, flow, name, hidden_dim=128):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        motion_features = BasicMotionEncoder(flow, corr, name='encoder')
        inp = tf.concat([inp, motion_features], axis=-1)

        net = SepConvGRU(net, inp, name='gru', hidden_dim=hidden_dim)
        delta_flow = FlowHead(net, name='flow_head', hidden_dim=256)

        with tf.variable_scope('mask'):
            m = Conv2D('0', net, 256, kernel_size=3, activation=tf.nn.relu)
            m = Conv2D('2', m, 64*9, kernel_size=1)
            mask = tf.multiply(0.25, m, name='scale_mask')

        return net, mask, delta_flow

def SmallUpdateBlock(net, inp, corr, flow, name, hidden_dim=96):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        motion_features = SmallMotionEncoder(flow, corr, name='encoder')
        inp = tf.concat([inp, motion_features], axis=-1)

        net = ConvGRU(net, inp, name='gru', hidden_dim=hidden_dim)
        delta_flow = FlowHead(net, name='flow_head', hidden_dim=128)
        return net, None, delta_flow


########################## corr #######################################
from networks.utils import bilinear_sampler
def GetCorrPyramid(fmap1, fmap2, num_levels=4):
    corr_pyramid = []
    with tf.name_scope('get_corr_pyramid'):
        with tf.name_scope('corr'):
            orig_shape = tf.shape(fmap1)
            h, w, c = orig_shape[1], orig_shape[2], orig_shape[3]

            fmap1 = tf.reshape(fmap1, tf.stack([-1, h*w, c]), name='reshape_fmap1')
            fmap2 = tf.reshape(fmap2, tf.stack([-1, h*w, c]), name='reshape_fmap2')

            fmap2_trans = tf.transpose(fmap2, (0, 2, 1)) # [-1, c, h*w]
            corr = tf.matmul(fmap1, fmap2_trans) # [-1, h1*w1, h2*w2]
            corr = tf.reshape(corr, tf.stack([-1, h, w, h, w, 1]))

            corr = tf.divide(corr, tf.sqrt(tf.cast(c, dtype=tf.float32)))

        corr = tf.reshape(corr, tf.stack([-1, h, w, 1])) # [b*h1*w1, h2, w2, 1]
        corr_pyramid.append(corr)
        for i in range(num_levels-1):
            corr = AvgPooling('pool2d{}'.format(i), corr, pool_size=2, strides=2)
            corr_pyramid.append(corr)

        return corr_pyramid


def SampleCorr(corr_pyramid, coords, num_levels=4, radius=4):
    # coords: [b,h,w,2]
    with tf.name_scope('sampling_radius_corr'):
        b = tf.shape(coords)[0]
        h = tf.shape(coords)[1]
        w = tf.shape(coords)[2]

        out_pyramid = []

        for i in range(num_levels):
            corr = corr_pyramid[i] # [b*h1*w1, h2, w2, 1]
            dx = tf.linspace(-1.0*radius, 1.0*radius, 2*radius+1)
            dy = tf.linspace(-1.0*radius, 1.0*radius, 2*radius+1)
            delta = tf.stack(tf.meshgrid(dy, dx)[::-1], axis=-1) # [2r+1,2r+1,2]

            centroid_lvl = tf.reshape(coords, (b*h*w, 1, 1, 2)) / (2**i) # [b*h*w, 1, 1, 2]
            delta_lvl = tf.reshape(delta, (1, 2*radius+1, 2*radius+1, 2)) # [b, 2r+1, 2r+1, 2]
            coords_lvl = centroid_lvl + delta_lvl # [b*h*w, 2r+1, 2r+1, 2]


            corr = bilinear_sampler(corr, coords_lvl) # [b*h1*w1, 2r+1, 2r+1, 1]
            corr = tf.reshape(corr, (b, h, w, -1)) # [b, h1, w1, (2r+1)**2]
            out_pyramid.append(corr)

        out = tf.concat(out_pyramid, axis=-1)
        return out

