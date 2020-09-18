import multiprocessing
import os
from tensorpack import logger
from tensorpack.callbacks import *
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.dataflow import (
    imgaug, dataset, AugmentImageComponent, AugmentImageComponents, PrefetchDataZMQ, BatchData, MapData, LMDBSerializer, PrefetchData,
    LocallyShuffleData, MultiThreadMapData)
import tensorflow as tf
import cv2
import numpy as np

class FlowDataProcess(object):
    def __init__(self, input_size, general_augmentation=False, rgb_augmentation=False, random_crop=False, test_mode=False):
        self.input_size = input_size
        assert isinstance(self.input_size, (list, tuple)) and len(self.input_size) == 2


        if general_augmentation:
            self.aug_gen = imgaug.AugmentorList([
                imgaug.Flip(horiz=True),
                 # imgaug.Rotation(15.0, (0.4, 0.6), border=cv2.BORDER_CONSTANT, border_value=[0, 0, 0])
            ])
        else:
            self.aug_gen = None


        if rgb_augmentation:
            self.aug_rgb = imgaug.AugmentorList([
                imgaug.Contrast((0.8, 1.2)),
                imgaug.Gamma((-0.3, 0.3)),
                imgaug.GaussianBlur(size_range=(0, 3), sigma_range=(0.2, 0.5)),
                imgaug.JpegNoise(quality_range=(70, 100))
            ])
        else:
            self.aug_rgb = None

        if random_crop:
            self.random_crop = imgaug.RandomCrop(crop_shape=input_size)
        else:
            self.random_crop = None

        self.test_mode = test_mode

    def __call__(self, x):
        np.random.seed()
        if self.aug_rgb is not None:
            self.aug_rgb.reset_state()
        if self.aug_gen is not None:
            self.aug_gen.reset_state()
        if self.random_crop is not None:
            self.random_crop.reset_state()


        # left image
        x[0] = cv2.imdecode(x[0], cv2.IMREAD_COLOR) # BGR
        # x[0] = cv2.cvtColor(x[0], cv2.COLOR_BGR2RGB) # RGB
        # print("read shape: ", x[0].shape)

        # right image
        x[1] = cv2.imdecode(x[1], cv2.IMREAD_COLOR)
        # x[1] = cv2.cvtColor(x[1], cv2.COLOR_BGR2RGB)


        # random switch order
        if np.random.choice([0, 1]) > 0:
            x[0], x[1] = x[1], x[0]

        height, width = x[0].shape[0: 2]
        if self.aug_rgb is not None: # TODO check input value range
            x[0], _prms = self.aug_rgb.augment_return_params(x[0])
            x[1] = self.aug_rgb.augment_with_params(x[1], _prms)


        if self.aug_gen is not None:
            x[0], _prms = self.aug_gen.augment_return_params(x[0])
            x[1] = self.aug_gen.augment_with_params(x[1], _prms)

        if self.random_crop is not None:
            x[0], _prms = self.random_crop.augment_return_params(x[0])
            x[1] = self.random_crop.augment_with_params(x[1], _prms)

        else:
            if self.test_mode:
                x[0] = cv2.resize(x[0], dsize=(self.input_size[1], self.input_size[0]))
                x[1] = cv2.resize(x[1], dsize=(self.input_size[1], self.input_size[0]))

            else:
                # evaluation when training
                x[0] = x[0][:self.input_size[0], :self.input_size[1], :]
                x[1] = x[1][:self.input_size[0], :self.input_size[1], :]



        x[0] = np.float32(x[0]) / 255.0 # 0~1
        x[1] = np.float32(x[1]) / 255.0

        return x

def get_test_dataflow(filelist, input_size):
    # filelist: [[im1_path, im2_path], [im2_path, im2_path], ...]
    from tensorpack.dataflow import DataFlow
    class Testset(DataFlow):
        def __init__(self, imglist):
            self.imglist = imglist
        def __len__(self):
            return len(self.imglist)
        def __iter__(self):
            idx = np.arange(len(self.imglist))
            for k in idx:
                pairs = self.imglist[k]
                with open(pairs[0], 'rb') as f1:
                    im1 = f1.read()
                    im1_bin = np.asarray(bytearray(im1), dtype='uint8')
                with open(pairs[1], 'rb') as f2:
                    im2 = f2.read()
                    im2_bin = np.asarray(bytearray(im2), dtype='uint8')

                yield [im1_bin, im2_bin]

    assert isinstance(input_size, (int, list, tuple)), input_size
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    ds = Testset(filelist)
    _parser = FlowDataProcess(input_size, test_mode=True)
    ds = MapData(ds, _parser)
    ds = BatchData(ds, 1)

    return ds