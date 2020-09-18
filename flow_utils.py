import numpy as np
import cv2
from copy import deepcopy


def make_colorwheel():
    '''
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    '''

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel

def flow_compute_color(u, v, convert_to_bgr=False):
    '''
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param u: np.ndarray, input horizontal flow
    :param v: np.ndarray, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:
    '''

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi #[h,w]

    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(np.int32)
    k0[k0 > 53] = 53
    k1 = k0 + 1
    k1[k1 == ncols] = 1
    f = fk - k0

    for i in range(colorwheel.shape[1]):

        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0 # [h,w]
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)

    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    '''
    Expects a two dimensional flow image of shape [H,W,2]
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param flow_uv: np.ndarray of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :return:
    '''

    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_compute_color(u, v, convert_to_bgr)

def aug_img(im):
    def usm_aug(im):
        im_g = cv2.GaussianBlur(im, (0,0), 5)
        usm = cv2.addWeighted(im, 1.5, im_g, -0.5, 0)
        return usm
    def contract_aug(im, a=1.5, b=0):
        im_aug = a * im + b
        im_aug = np.clip(im_aug, 0, 255)
        return np.uint8(im_aug)

    im = contract_aug(im)
    im = usm_aug(im)
    return im

def calc_flow(im0, im1, use_yuv=False):
    if not use_yuv:
        im0_gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    else:
        im0_yuv = cv2.cvtColor(im0, cv2.COLOR_BGR2YUV)
        im1_yuv = cv2.cvtColor(im1, cv2.COLOR_BGR2YUV)
        im0_gray = im0_yuv[:, :, 0]
        im1_gray = im1_yuv[:, :, 0]

    inst = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = inst.calc(im0_gray, im1_gray, None)

    # post-process
    flow_post = cv2.ximgproc.guidedFilter(im0, flow, radius=9, eps=2)

    return flow_post

def set_static_flow(flow01_origin, im0, bg):
    static_diff_im0 = np.abs(bg - im0)
    static_mask = np.prod(static_diff_im0<5, axis=-1, keepdims=True)
    flow01 = np.where(static_mask, 0.0, flow01_origin)
    return flow01

def erode5(mask, r=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (r, r))
    return cv2.erode(mask, kernel)

# skip flow value=INF
def reverse_flow_avg_skip_static(flow01_origin, bg, im0, time_step=1.0):
    static_diff_im0 = np.abs(bg - im0)
    static_mask = np.prod(static_diff_im0<10, axis=-1, keepdims=True) # h,w,1
    flow01 = np.where(static_mask, np.inf, flow01_origin)

    h, w = flow01.shape[0:2]
    flow01 = flow01 * time_step

    flowx = flow01[:, :, 0] + np.arange(w) # new_position
    flowy = flow01[:, :, 1] + np.arange(h)[:, np.newaxis]

    flow10 = np.zeros_like(flow01)
    conflict_count = np.zeros((h, w))

    FLOW_PROJECTION_ROUND = True

    for y in range(h): # for each position in flow01
        for x in range(w):
            new_x_pos = flowx[y, x]
            new_y_pos = flowy[y, x]

            if np.isinf(new_x_pos) or np.isinf(new_y_pos):
                continue

            if FLOW_PROJECTION_ROUND:
                round_x_pos = np.clip(int(np.round(new_x_pos)), 0, w-1)
                round_y_pos = np.clip(int(np.round(new_y_pos)), 0, h-1)
                flow10[round_y_pos, round_x_pos, :] += -1.0 * flow01[y, x, :]
                conflict_count[round_y_pos, round_x_pos] += 1

            else:
                bilinear_x_pos = [np.floor(new_x_pos), np.ceil(new_x_pos), np.ceil(new_x_pos), np.floor(new_x_pos)]
                bilinear_y_pos = [np.floor(new_y_pos), np.floor(new_y_pos), np.ceil(new_y_pos), np.ceil(new_y_pos)]

                # TODO: directly round(new_x_pos) instead of bilinear inter
                for i in range(4):
                    int_x_pos = int(bilinear_x_pos[i])
                    int_y_pos = int(bilinear_y_pos[i])

                    if int_x_pos < 0 or int_x_pos >w-1:
                        continue
                    if int_y_pos<0 or int_y_pos>h-1:
                        continue

                    dist_weight = np.sqrt((new_x_pos - int_x_pos)**2 + (new_y_pos - int_y_pos)**2)
                    # print("int ", int_x_pos, int_y_pos)
                    flow10[int_y_pos, int_x_pos, :] += -dist_weight*flow01[y, x, :]
                    conflict_count[int_y_pos, int_x_pos] += dist_weight

    flow10[:,:,0] = np.where(conflict_count>1e-7, flow10[:,:,0] / conflict_count, flow10[:,:,0])
    flow10[:,:,1] = np.where(conflict_count>1e-7, flow10[:,:,1] / conflict_count, flow10[:,:,1])
    empty = np.uint8(conflict_count<=1e-7)
    empty_before_static = deepcopy(empty)


    print(" {} pixels are empty before fill static pixels.".format(np.sum(empty)))
    empty_before_static = erode5(empty_before_static)
    # flow10[:, :, 0] = np.where(np.logical_and(empty_before_static, static_mask[:, :, 0]), 0.0, flow10[:, :, 0])
    # flow10[:, :, 1] = np.where(np.logical_and(empty_before_static, static_mask[:, :, 0]), 0.0, flow10[:, :, 1])
    # conflict_count = np.where(np.logical_and(empty_before_static, static_mask[:, :, 0]), 1, conflict_count)
    empty = np.uint8(conflict_count<=1e-7)
    print(" {} pixels are empty after fill static pixels with 0.".format(np.sum(empty)))

    def fiil_ind(indy, indx):
        nearest_count = 0
        # up
        nn_flow_sum = [0, 0]
        for y in range(indy-1, -1, -1):
            if empty[y, indx] == 0:
                # not empty
                nn_flow_sum[0] += flow10[y, indx, 0]
                nn_flow_sum[1] += flow10[y, indx, 1]
                nearest_count += 1
                break
        for y in range(indy, h):
            if empty[y, indx] == 0:
                nn_flow_sum[0] += flow10[y, indx, 0]
                nn_flow_sum[1] += flow10[y, indx, 1]
                nearest_count += 1
                break
        for x in range(indx-1, -1, -1):
            if empty[indy, x] == 0:
                nn_flow_sum[0] += flow10[indy, x, 0]
                nn_flow_sum[1] += flow10[indy, x, 1]
                nearest_count += 1
                break
        for x in range(indx, w):
            if empty[indy, x] == 0:
                nn_flow_sum[0] += flow10[indy, x, 0]
                nn_flow_sum[1] += flow10[indy, x, 1]
                nearest_count += 1
                break
        if nearest_count == 0:
            flow10[indy, indx, :] = 0
            return
        flow10[indy, indx, 0] = nn_flow_sum[0] / nearest_count
        flow10[indy, indx, 1] = nn_flow_sum[1] / nearest_count

    empty_ind_arrays = np.where(empty)
    empty_num = len(empty_ind_arrays[0])
    for i in range(empty_num):
        indy = empty_ind_arrays[0][i]
        indx = empty_ind_arrays[1][i]
        fiil_ind(indy, indx)
        # empty[indy, indx] = 0
        # print("fill in ", indy, indx)


    return flow10, empty, np.uint8(conflict_count > 1), static_mask, empty_before_static


def resize_flow(flow, dW, dH):
    H_, W_ = flow.shape[0:2]
    u_ = cv2.resize(flow[:, :, 0], (dW, dH))
    v_ = cv2.resize(flow[:, :, 1], (dW, dH))
    u_ *= dW / float(W_)
    v_ *= dH / float(H_)

    return np.dstack((u_, v_))

def readFlow(name):
    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height,
                                                                   width, 2))
    f.close()
    return flow.astype(np.float32)


def writeFlow(flow, filename):
    """Write optical flow to file

    Args:
        flow(ndarray): optical flow
        filename(str): file path
        quantize(bool): whether to quantize the flow and save as 2 images,
                        if set to True, remaining args will be passed
                        to :func:`quantize_flow`
    """

    with open(filename, 'wb') as f:
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()