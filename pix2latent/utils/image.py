from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import numpy as np
import cv2
import os
import torchvision
from PIL import Image
from torchvision import transforms


def read(im_path, as_transformed_tensor=False, im_size=512, transform_style=None):
    im = np.array(Image.open(im_path).convert("RGB"))
    h, w = im.shape[:2]

    if np.max(im) <= 1. + 1e-6:
        im = (im * 255).astype(np.uint8)

    im = Image.fromarray(im)

    if as_transformed_tensor:

        if transform_style == 'biggan':
            transform = transforms.Compose(
                [
                    transforms.Resize(im_size),
                    transforms.CenterCrop(im_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

        elif transform_style in ['stylegan', 'stylegan2']:
            if h < w:
                pad_top = (w - h) // 2
                pad_bot = w - h - pad_top
                pad_left, pad_right = 0, 0
            else:
                pad_left = (h - w) // 2
                pad_right = h - w - pad_left
                pad_top, pad_bot = 0, 0

            transform = transforms.Compose([
                    transforms.Pad((pad_left, pad_top, pad_right, pad_bot)),
                    transforms.Resize(im_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    ])

        elif transform_style is None:
            transform = transforms.Compose([
                    transforms.Resize(im_size),
                    transforms.CenterCrop(im_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    ])

        else:
            raise ValueError(f'unknown transformation style {transform_style}')

    return transform(im)


def save(save_path, im):
    if type(im) is torch.Tensor: #np.ndarray: # assume torch
        im = to_image(im, cv2_format=False)
    return cv2.imwrite(save_path, im[:, :, [2,1,0]],
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def to_grid(x):
    grid_sz = int(np.ceil(np.sqrt(x.size(0))))
    return torchvision.utils.make_grid(x, grid_sz, pad_value=-1)


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
    mask = torch.from_numpy(mask).permute(2, 0, 1)
    return torch.clamp(mask.unsqueeze(0).cuda().float(), 0.0, 1.0)


def binarize(mask, min=0.0, max=1.0, eps=1e-3):
    """ used to convert continuous valued mask to binary mask """
    if type(mask) is torch.Tensor:
        assert mask.max() <= 1 + 1e-6, mask.max()
        assert mask.min() >= -1 - 1e-6, mask.min()
        mask = (mask > 1.0 - eps).float()
        return mask.clamp_(min, max)
    elif type(mask) is np.ndarray:
        mask = (mask > 1.0 - eps).astype(float)
        return np.clip(mask, min, max, out=mask)
    return False


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
    from transform.transform_utils import compute_stat_from_mask

    if np.max(target) <= 1.0:
        target = target * 255.
    if np.max(generated) <= 1.0:
        generated = generated * 255.
    if np.max(mask) > 1.0:
        mask = mask / 255.

    obj_center, _ = compute_stat_from_mask(
        binarize(torch.Tensor(mask).permute(2, 0, 1)))

    obj_center = (int(obj_center[1] * target.shape[1]), \
                  int(obj_center[0] * target.shape[0]))

    mask = (mask > 0.5).astype(np.float)

    blended_result = cv2.seamlessClone(
                                generated.astype(np.uint8),
                                target.astype(np.uint8),
                                (255 * mask[:, :, 0]).astype(np.uint8),
                                obj_center,
                                cv2.NORMAL_CLONE
                                )

    return blended_result
