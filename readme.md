This repository contains the tf-implementation(with tensorpack) for this paper:

[RAFT:Recurrent All Pairs Field Transforms for Optifal Flow](https://arxiv.org/pdf/2003.12039.pdf)

[Official Pytorch Implementation](https://github.com/princeton-vl/RAFT)

## TODO
* [x] Basic inference code.
* [ ] grid_sample `align_corners=True` in tf-implementation.
* [ ] Add cuda extension for efficent correlation calculation.
* [ ] Check if the batch_size could be free.
* [ ] Reproduce the training process.

## Requirements

```
tensorflow-gpu >= 1.14
opencv-python
numpy
tensorpack
CUDA 10.1
```

## Demo

1. Download the pretrained model from [GoogleDrive]() to `release_weight` folder. The *.npz files are converted from the official pytorch *.pth model provided in the [official repository](https://github.com/princeton-vl/RAFT).

2. Run the inference demo:
```
bash ./infer_image.sh
```

or

```
python infer_raft.py --im1 frame_0016.png --im2 frame_0017.png --load release_weight/raft-things.npz

python infer_raft.py --im1 frame_0016.png --im2 frame_0017.png --load release_weight/raft-small.npz --small

```

## Note
The inference result  of my tensorflow implementation is as below. There is still a few of differences from the official implementation. I will continue to follow up.

| My tf-implementation | Official pytorch-implementation |
| ---- | ---- |
| ![my](./raft_flow_raft-things.png) | ![offcial]() |