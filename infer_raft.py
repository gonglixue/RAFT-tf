import os
import argparse
import multiprocessing
import lmdb
import cv2
import numpy as np
import datetime

# tf
import tensorflow as tf
from tensorpack import logger, QueueInput, StagingInput, FeedInput
from tensorpack.callbacks import *
from tensorpack.train import (TrainConfig, SyncMultiGPUTrainerParameterServer,
                              launch_train_with_config, SimpleTrainer)
from tensorpack.tfutils import argscope, get_model_loader, model_utils
from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.predict import PredictConfig, FeedfreePredictor
from tensorpack import *

from networks import RAFT
from dataflow import test_dataflow

def inference_anonymous(model, sessinit, outpath, dataflow):
    from flow_utils import flow_to_color
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    output_names = ['flow_result']

    pred_config = PredictConfig(
        model=model,
        session_init=sessinit,
        input_names=['input_left', 'input_mid', 'input_right'],
        output_names=output_names
    )
    pred = FeedfreePredictor(pred_config, StagingInput(QueueInput(dataflow), device='/gpu:0'))

    for _id in range(dataflow.size()):
        _preds = pred()

        print(_id, _preds[0].shape)
        flow_color = flow_to_color(_preds[0][0], convert_to_bgr=True)
        cv2.imwrite("raft_flow_raft-things.png", flow_color)
        cv2.imshow("raft_flow", flow_color)
        cv2.waitKey(0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='1',
                        help='comma separated list of GPU(s) to use. Default to use all available ones')
    parser.add_argument('--data', default='/mnt/cephfs_new_wj/ies/gonglixue/dataset/vimeo_triplet_lmdb/vimeo_triplet',
                        help='Path to LMDB dataset')
    parser.add_argument('--load', default='release_weight/raft-things.npz', help='load a model for training or evaluation')
    parser.add_argument('-m', '--mode', help='Run the network in training / evaluation / test mode',
                        default='test', choices=['train', 'val', 'test', 'export', 'flops'])
    parser.add_argument('--out', default='./log',
                        help='output path for evaluation and test, default to current folder')
    parser.add_argument('--batch', default=1, type=int, help="Batch size per tower.")
    parser.add_argument('-o', '--optimizer', help='Optimizer used in training.',
                        default='adam', choices=['adam', 'adamw', 'sgd', 'sgd_cyclic', 'sgd_1cycle'])
    parser.add_argument('--im1', help='left image path.', default='frame_0010.png')
    parser.add_argument('--im2', help='right image path.', default='frame_0011.png')
    parser.add_argument('--small', action='store_true')
    args = parser.parse_args()

    model = RAFT.RAFT((432, 1024, 3), args)

    if args.mode == 'export':
        pass

    if args.mode == 'test':
        filelist = [[args.im1, args.im2]]
        ds = test_dataflow.get_test_dataflow(filelist, (432, 1024))
        sess_init = get_model_loader(args.load)
        inference_anonymous(model, sess_init, 'output_raft', ds)

    if args.mode == 'flops':
        with TowerContext('', is_training=False):
            model.build_clean_graph(
                tf.placeholder(tf.float32, [1, 256, 448, 3], 'input_left'),
                tf.placeholder(tf.float32, [1, 256, 448, 3], 'input_right')
            )
            model_utils.describe_trainable_vars()

            flops = tf.profiler.profile(
                tf.get_default_graph(),
                cmd='op',
                options=tf.profiler.ProfileOptionBuilder.float_operation())
            print("total flops: ", flops.total_float_ops)
            logger.info("Note that TensorFlow counts flops in a different way from the paper.")
            logger.info("TensorFlow counts multiply+add as two flops, however the paper counts them "
                        "as 1 flop because it can be executed in one instruction.")