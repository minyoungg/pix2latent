from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import numpy as np
import warnings
import cv2
import os
import imageio
import skvideo.io
import torchvision


make_grid = lambda x : \
        torchvision.utils.make_grid(x, int(np.sqrt(x.size(0))), pad_value=-1)


def to_image(output, to_cpu=True, denormalize=True, jpg_format=True,
             to_numpy=True, cv2_format=True):
    """ Formats torch tensor in the form BCHW -> BHWC """
    is_batched = True
    if len(list(output.size())) == 3:
        output = output.unsqueeze(0)
        is_batched = False
    tmp = output.detach().float()
    if to_cpu:
        tmp = tmp.cpu()

    tmp = tmp.permute(0, 2, 3, 1)
    if denormalize:
        tmp = (tmp + 1.0) / 2.0
    if jpg_format:
        tmp = (tmp * 255).int()
    if cv2_format and output.size(1) > 1:
        tmp = tmp[:, :, :, [2, 1, 0]]
    if to_numpy:
        tmp = tmp.numpy()
    if not is_batched:
        return tmp.squeeze(0)
    return tmp


def to_tensor(im):
    if type(im) == str:
        im = cv2.imread(im)[:, :, [2, 1, 0]]
    if np.max(im) > 1:
        im = im / 255.
    im = (2.0 * (torch.from_numpy(im).float() - 0.5)).permute(2, 0, 1)
    return im.unsqueeze(0).cuda()


def to_mask(mask):
    """
    Reads mask images and formats to tensor mask of dimension [1, H, W]
    in range(0, 1)
    """
    if type(mask) == str:
        assert os.path.exists(mask)
        mask = (cv2.imread(mask)[:, :, :1] / 255. > 0.5).astype(np.float)

    assert np.max(mask) <= 1.0 and np.min(mask) >= 0.0
    mask = 1.0 - torch.from_numpy(mask).permute(2, 0, 1)
    return torch.clamp(mask.unsqueeze(0).cuda().float(), 0.0, 1.0)


def binarize(mask, min=0.0, max=1.0, eps=1e-3):
    """ used to convert continuous valued mask to binary mask """
    assert mask.max() <= 1 + 1e-6, mask.max()
    assert mask.min() >= -1 - 1e-6, mask.min()
    mask = (mask > 1.0 - eps).float()
    return torch.clamp(mask, min, max)


def vis_gif(save_path, ims, duration=20.0):
    """ dumps a list of images into gif """
    dpf = duration / len(ims) # limited to 12fps
    imageio.mimsave(save_path, tracked_ims, duration=dpf)
    return


def add_border(image, color='r', border_size=5, cv2_format=True):
    """ adds border around an image """
    if cv2_format:
        r, g, b  = [[0, 0, 255]], [[0, 255, 0]], [[255, 0, 0]]
    else:
        r, g, b = [[255, 0, 0]], [[0, 255, 0]], [[0, 0, 255]]

    if color == 'r':
        clr = r
    elif color == 'g':
        clr = g
    elif color == 'b':
        clr = b
    else:
        raise ValueError('Unknown color {}'.format(r))

    image[:border_size] = clr
    image[-border_size:] = clr
    image[:, :border_size] = clr
    image[:, -border_size:] = clr
    return image


def make_video(save_path, ims, fps=30, duration=None, safe=True):
    """
    Creates a video given an array of images. Uses FFMPEG backend.
    Depending on the FFMPEG codec supported you might have to change pix_fmt
    To change the quality of the saved videos, change -b (the encoding bitrate)
    Args:
      save_path: path to save the video
      ims: an array of images
      fps: frames per seconds to save the video
      duration: the duration of the video, if not None, will override fps.

    > ims = [im1, im2, im3]
    > make_video('video.mp4', ims, fps=10)
    """
    if np.max(ims) <= 1:
        ims = (np.array(ims) * 255).astype(np.uint8)
    if duration is not None:
        fps = len(ims) / duration

    skvideo.io.vwrite(save_path, ims,
                      inputdict={'-r': str(fps)},
                      outputdict={'-r': str(fps),
                                  '-pix_fmt': 'yuv420p',
                                  '-b': '10000000'})
    return


def center_crop(image):
    """
    Center crop images along the maximum dimension
    Args:
        image:
            A numpy array of an image HW3
    Returns:
        cropped_image: A cropped image with H == W
    """
    h, w = image.shape[:2]
    if h > w:
        h_st = (h - w) // 2
        h_en = h_st + w
        cropped_image = image[h_st: h_en, :, :]
    else:
        w_st = (w - h) // 2
        w_en = w_st + h
        cropped_image = image[:, w_st: w_en, :]
    assert cropped_image.shape[0] == cropped_image.shape[1]
    return cropped_image


def smart_resize(im, target_size=(256, 256)):
    """
    Resize an image into target spatial dimension. If resizing to a smaller
    area, uses cv2.INTER_AREA else cv2.INTER_BILINEAR
    """
    if np.prod(im.shape[:2]) >= np.prod(target_size):
        interp_fn = cv2.INTER_AREA
    else:
        interp_fn = cv2.INTER_LINEAR
    return cv2.resize(im, (target_size[1], target_size[0]),
                      interpolation=interp_fn)


def poisson_blend(target, mask, generated):
    from transform_utils import compute_stat_from_mask

    if np.max(target) <= 1.0:
        target = target * 255.
    if np.max(generated) <= 1.0:
        generated = generated * 255.
    if np.max(mask) > 1.0:
        mask = mask / 255.

    obj_center, _ = compute_stat_from_mask(
                binarize(torch.Tensor(mask).permute(2, 0, 1).unsqueeze(0)))

    mask = (mask > 0.5).astype(np.float)

    blended_result = cv2.seamlessClone(
                                    generated.astype(np.uint8),
                                    target.astype(np.uint8),
                                    (255 *  mask[:, :, 0]).astype(np.uint8),
                                    obj_center[::-1],
                                    cv2.NORMAL_CLONE
                                    )

    return blended_result
