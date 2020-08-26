from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np

import torch

import pix2latent.transform.transform_functions as TF
from pix2latent.utils.image import binarize


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


def convert_to_t(src_center, src_size, dst_center, dst_size):
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
    dxy = (src_center - dst_center) * 2.
    t = np.array([s, *dxy[::-1]])
    return torch.from_numpy(t).float()


def get_biggan_stats():
    """ precomputed biggan statistics """
    center_of_mass = [137 / 255., 127 / 255.]
    object_size = [213 / 255., 210 / 255.]
    return center_of_mass, object_size


def compute_stat_from_mask(mask):
    """ Given a binarized mask 0, 1. Compute the object size and center """
    st_h, st_w, en_h, en_w = bbox_from_mask(mask)
    obj_size = obj_h, obj_w = en_h - st_h, en_w - st_w
    obj_center = (st_h + obj_h // 2, st_w + obj_w // 2)

    obj_size = (obj_size[0] / mask.size(1), obj_size[1] / mask.size(2))
    obj_center = (obj_center[0] / mask.size(1), obj_center[1] / mask.size(2))
    return obj_center, obj_size


def bbox_from_mask(mask):
    assert len(list(mask.size())) == 3, \
        'expected 3d tensor but got {}'.format(len(list(mask.size())))

    try:
        tlc_h = (mask.mean(0).sum(1) != 0).nonzero()[0].item()
        brc_h = (mask.mean(0).sum(1) != 0).nonzero()[-1].item()
    except:
        tlc_h, brc_h = 0, mask.size(1)  # max range if failed

    try:
        tlc_w = (mask.mean(0).sum(0) != 0).nonzero()[0].item()
        brc_w = (mask.mean(0).sum(0) != 0).nonzero()[-1].item()
    except:
        tlc_w, brc_w = 0, mask.size(2)
    return tlc_h, tlc_w, brc_h, brc_w




class ComposeTransform():
    """
    Composes multiple transform function.
    Transform list is consists of either transform function or a tuple with
    the second element corresponding to the weight. The weight are used to
    scale the t_parameter as eact parameters will live at a different scale.
    If the weights are not provided, default weight of 1 will be used.
    """

    def __init__(self, transform_list):
        assert type(transform_list) == list
        self.transform_list = []
        for t_fn in transform_list:
            if type(t_fn) in [tuple, list]:
                self.transform_list.append(t_fn)
            else:
                self.transform_list.append([t_fn, 1.0])

        self._t = np.array([x[0].t for x in self.transform_list])
        return

    def get_param(self, as_tensor=False):
        """ Returns the default parameter set during intialization"""
        if as_tensor:
            return torch.Tensor(np.concatenate(self._t))
        return self._t

    def get_opt_param(self):
        """ Returns optimize-able parameters """
        _x = [x[0].get_opt_param() for x in self.transform_list]
        return np.concatenate(_x)

    def reweight(self, t, weight, t_mean):
        """ Scales the transformation parameter after mean normalization """
        return (weight * (t - t_mean)) + t_mean

    def __call__(self, ims, t, invert=False, only_spatial=False):
        """
        Applies transformation to the images.
        Note
            Transformations are inverted in the same order it was applied.
            Operations may be order dependent and so maybe we should apply it
            in the reverse order.
        """

        if t.size(0) == 1:
            t = t.repeat(ims.size(0), 1)

        t_i = 0
        for i, (fn, fn_weight) in enumerate(self.transform_list):
            t_sz = len(fn.t)
            if (only_spatial and fn.is_spatial) or not only_spatial:
                t_param = t[:, t_i:t_i + t_sz]
                t_mu = torch.from_numpy(self._t[i]).type_as(t_param)
                t_param = self.reweight(t_param, fn_weight, t_mu)
                ims = fn(ims, t_param, invert=invert)
            t_i += t_sz
        return ims

    def __str__(self):
        fmt = '<ComposeTransform\n\t{}\n>'.format(
            '\n\t'.join([f[0].__str__() for f in self.transform_list]))
        return fmt
