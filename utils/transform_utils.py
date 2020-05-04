from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np

import torch

import transform_functions as TF
from im_utils import binarize


def setup_transform_fn(args, weight):
    transform_list = []
    transform_fn, t = None, None

    # -- Affine transformation -- #
    if args.spatial_transform:
        transform_list += [(TF.SpatialTransform(optimize=True), 1.0)]
    else:
        if args.align:
            transform_list += [(TF.SpatialTransform(optimize=False), 1.0)]

    # -- Color transformation -- #
    # Ordered by information preservability

    if 'hue' in args.color_transform:
        transform_list += [(TF.HueTransform(), 5.0)]
    if 'gamma' in args.color_transform:
        transform_list += [(TF.GammaTransform(), 5.0)]
    if 'saturation' in args.color_transform:
        transform_list += [(TF.SaturationTransform(), 5.0)]
    if 'brightness' in args.color_transform:
        transform_list += [(TF.BrightnessTransform(), 5.0)]
    if 'contrast' in args.color_transform:
        transform_list += [(TF.ContrastTransform(), 5.0)]

    # Compose multiple transformations
    if len(transform_list) > 0:
        transform_fn = TF.ComposeTransform(transform_list)
        t = transform_fn.get_param()

        # Override with pre-alignmnet
        if args.align:
            t[0] = compute_pre_alignment(weight)

        t = torch.from_numpy(np.concatenate(t)).unsqueeze(0).float()
    return transform_fn, t


def compute_pre_alignment(weight):
    """ Precompute intialization based on BigGAN bias """
    dst_center, dst_size = get_biggan_stats()
    src_center, src_size = compute_stat_from_mask(binarize(weight))
    t = convert_to_t(src_center, src_size, dst_center, dst_size)
    return t.numpy()


def convert_to_t(src_center, src_size, dst_center, dst_size, im_size=256):
    """
    Given the object center and target size, returns transformation parameter
    t that will transform the object to the dst_center and dst_size
    Args:
        src_center: (tuple) current object center
        src_size: (tuple) current object size
        dst_center: (tuple) target object center
        dst_size: (tuple) target object size
    Returns:
        t: transformation parameter
    """

    src_center, src_size = np.array(src_center), np.array(src_size)
    dst_center, dst_size = np.array(dst_center), np.array(dst_size)

    scale_idx = np.argmax(src_size).squeeze()
    s = (src_size / dst_size)[scale_idx]
    dxy = (src_center - dst_center) / (im_size / 2.)
    t = np.array([s, *dxy[::-1]])
    return torch.from_numpy(t).float()


def get_biggan_stats():
    """ precomputed biggan statistics """
    center_of_mass = [137, 127]
    object_size = [213, 210]
    return center_of_mass, object_size


def compute_stat_from_mask(mask):
    """ Given a binarized mask 0, 1. Compute the object size and center """
    st_h, st_w, en_h, en_w = bbox_from_mask(mask)
    obj_size = obj_h, obj_w = en_h - st_h, en_w - st_w
    obj_center = (st_h + obj_h // 2, st_w + obj_w // 2)
    return obj_center, obj_size


def bbox_from_mask(mask):
    assert len(list(mask.size())) == 4, \
        'expected 4d tensor but got {}'.format(len(list(mask.size())))
    try:
        tlc_h = (mask.squeeze().sum(1) != 0).nonzero()[0].item()
        brc_h = (mask.squeeze().sum(1) != 0).nonzero()[-1].item()
    except:
        tlc_h, brc_h = 0, mask.size(2)  # max range if failed

    try:
        tlc_w = (mask.squeeze().sum(0) != 0).nonzero()[0].item()
        brc_w = (mask.squeeze().sum(0) != 0).nonzero()[-1].item()
    except:
        tlc_w, brc_w = 0, mask.size(3)
    return tlc_h, tlc_w, brc_h, brc_w
