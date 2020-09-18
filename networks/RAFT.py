import tensorflow as tf
from tensorpack.tfutils import argscope, varreplace
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack import ModelDesc, logger
from tensorpack.models import *
from tensorpack.tfutils.summary import (add_moving_summary)
from tensorpack.tfutils import optimizer, gradproc

from networks.model_utils import *
from networks.utils import coords_grid, upflow8

class RAFT(ModelDesc):
    # input BGR to first layer
    weight_decay = 1e-5
    data_format = 'NHWC'  # export to frozen graph only support NHWC
    """
    To apply on normalization parameters, use '.*/W|.*/gamma|.*/beta'
    """
    regularization_pattern = '.*/W|.*/gamma|.*/beta|.*/b$'

    """
    training config
    """
    debug_param_summary = False

    def __init__(self, image_shape, args):
        # not small
        self.dropout = 0.0
        self.corr_radius = 4
        self.hidden_dim = 128
        self.context_dim = 128
        self.mode = 'test'
        self.iters = 20
        self.image_shape = image_shape
        self.batch = 1

        self.small = args.small
        if self.small:
            self.hidden_dim = 96
            self.context_dim = 64
            self.corr_radius = 3
        logger.info("Buil model with tensorflow-{}".format(tf.__version__))
        logger.info("is small: {}".format(self.small))

    def inputs(self):
        # TODO: check if `batch_size` can be free
        return [tf.placeholder(tf.float32, [self.batch, self.image_shape[0], self.image_shape[1], self.image_shape[2]],
                               'input_left'),
                tf.placeholder(tf.float32, [self.batch, self.image_shape[0], self.image_shape[1], self.image_shape[2]],
                               'input_right')
                ]

    def input_preprocess(self, input_left, input_right):
        # 0~1 -> -1,1
        with tf.name_scope('input_preprocess'):
            input_left = 2.0 * input_left - 1.0
            input_right = 2.0 * input_right - 1.0

            return input_left, input_right


    def feature_extractor(self, image):
        if self.small:
            fmap = SmallEncoder(image, name='fnet', out_dim=128, norm_fn='instance', dropout=self.dropout)
        else:
            fmap = BasicEncoder(image, name='fnet', output_dim=256, norm_fn='instance', dropout=self.dropout)
        return fmap

    def context_net(self, image):
        if self.small:
            ctx = SmallEncoder(image, name='cnet', out_dim=self.hidden_dim + self.context_dim, norm_fn='none',
                               dropout=self.dropout)
        else:
            ctx = BasicEncoder(image, name='cnet', output_dim=self.hidden_dim + self.context_dim, norm_fn='batch',
                               dropout=self.dropout)
        return ctx

    def network_graph(self, input_left, input_right):
        fmap1 = self.feature_extractor(input_left)
        fmap2 = self.feature_extractor(input_right)

        corr_pyramid = GetCorrPyramid(fmap1, fmap2)

        cnet = self.context_net(input_left)
        net, inp = tf.split(cnet, [self.hidden_dim, self.context_dim], axis=-1)
        net = tf.tanh(net)
        inp = tf.nn.relu(inp)

        coords0, coords1 = self.initialize_flow(input_left)

        for iter in range(self.iters):
            # TODO: detach?
            coords1 = tf.stop_gradient(coords1)
            corr = SampleCorr(corr_pyramid, coords1, radius=self.corr_radius)
            flow = coords1 - coords0 # [b,h,w,2]
            if self.small:
                net, up_mask, delta_flow = SmallUpdateBlock(net, inp, corr, flow, name='update_block',
                                                            hidden_dim=self.hidden_dim)
            else:
                net, up_mask, delta_flow = BasicUpdateBlock(net, inp, corr, flow, name='update_block',
                                                            hidden_dim=self.hidden_dim)
            coords1 = coords1 + delta_flow

        if self.small:
            flow_up = upflow8(coords1 - coords0)
        else:
            flow_up = self.upsample_flow(coords1 - coords0, up_mask)
        # flow_up = coords1 - coords0
        return flow_up

    def initialize_flow(self, image):
        with tf.name_scope('initialize_flow'):
            img_shape = image.get_shape().as_list()
            coords0 = coords_grid(tf.shape(image)[0], img_shape[1]//8, img_shape[2]//8) # coordinates_old [b,h,w,2]
            coords1 = coords_grid(tf.shape(image)[0], img_shape[1]//8, img_shape[2]//8)

            return coords0, coords1

    def upsample_flow(self, flow, mask):
        # up_mask: [b, h, w, 64*9]
        with tf.name_scope('upsample_flow'):
            h = tf.shape(flow)[1]
            w = tf.shape(flow)[2]
            b = tf.shape(flow)[0]
            mask = tf.reshape(mask, (-1, h, w, 9, 1, 8, 8)) # [b,h,w, 9, 1, 8, 8]
            mask = tf.nn.softmax(mask, axis=3)

            up_flow = tf.extract_image_patches(8*flow, ksizes=[1,3,3,1], strides=[1,1,1,1], rates=[1,1,1,1], padding='SAME') # [b,h,w,k*k*c]
            up_flow = tf.reshape(up_flow, (b, h, w, 9, 2, 1, 1))

            up_flow = tf.reduce_sum(up_flow * mask, axis=3) # [b,h,w,2,8,8]
            up_flow = tf.transpose(up_flow, (0, 1,4,2,5,3)) # [b,h,8,w,8,2]
            up_flow = tf.reshape(up_flow, (b, h*8, w*8, 2))
        return up_flow

    def build_graph(self, input_left, input_right):
        input_left, input_right = self.input_preprocess(input_left, input_right)

        flow_result = self.network_graph(input_left, input_right)
        flow_result = tf.identity(flow_result, name='flow_result')
        return 0.0

    def build_clean_graph(self, input_left, input_right):
        flow_result = self.network_graph(input_left, None, input_right)

        return 0.0