from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF


## --- Compose transformations --- ##

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

    def get_param(self):
        """ Returns the default parameter set during intialization"""
        return self._t

    def get_opt_param(self):
        """ Returns optimize-able parameters """
        _x = [x[0].get_opt_param() for x in self.transform_list]
        return np.concatenate(_x)

    def reweight(self, t, weight, t_mean):
        """ Scales the transformation parameter after mean normalization """
        return (weight * (t - t_mean)) + t_mean

    def __call__(self, ims, t=None, invert=False, only_spatial=False):
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


## --- COLOR TRANSFORMATIONS --- #

# Since lambda functions cant be pickled easily.
def _negate(x):
    return -x


def _invert(x):
    return 1.0 / x


class ColorTransform(object):
    """
    Base class for color transformations. Any function that inherits this class
    is not differentiable. A differentiable version could be implemented but
    since we use BasinCMA we use the default PyTorch/PIL color transfomration.
    """

    def __init__(self, fn, t=[1], t_range=(0.667, 1.5), t_inv_fn=None,
                 optimize=True):
        """
        Args:
            fn: Torchvision color transformation function
            t: Default starting parameter. This is used for initializing search
            t_range: A tuple that limits the transformation range (min, max)
            optimize: If trainable you returns the parameter
        """
        assert t_range[1] > t_range[0], 't_range should be increasing'
        self.fn = fn
        self.t = np.array(t, dtype=np.float32)
        self.t_inv_fn = t_inv_fn
        self.t_min, self.t_max = t_range
        self.is_spatial = False
        self.optimize = optimize
        return

    def get_opt_param(self):
        if self.optimize:
            return self.t
        return []

    def apply(self, ims, t, invert=False):
        """ Applies transformation fn(im, t) -- NOT DIFFERENTIABLE """
        assert ims.size(0) == t.size(0)
        assert t.size(1) == 1

        if invert:
            t = self.t_inv_fn(t)

        # Clamp by range
        t = torch.clamp(t, self.t_min, self.t_max)

        # Map [-1, 1] to [0, 1] and convert to PIL
        ims = (ims.detach().cpu() + 1.0) / 2.0
        ims = [TVF.to_pil_image(im) for im in ims]

        # Apply torchvision transform and convert it back to tensor
        ims = [self.fn(im, _t.item()) for im, _t in zip(ims, t)]
        ims = 2.0 * (torch.stack([TVF.to_tensor(im) for im in ims]) - 0.5)
        return ims.float().cuda()

    def __call__(self, ims, t, invert=False):
        return self.apply(ims, t, invert)

    def __str__(self):
        return 'ColorTransform: {}'.format(self.fn)


class HueTransform(ColorTransform):
    def __init__(self, t=[0], t_min=-0.5, t_max=0.5):
        super().__init__(fn=TVF.adjust_hue,
                         t=t,
                         t_range=(t_min + 1e-6, t_max - 1e-6),
                         t_inv_fn=_negate)
        return


class BrightnessTransform(ColorTransform):
    def __init__(self, t=[1], t_min=0.667, t_max=1.5):
        super().__init__(fn=TVF.adjust_brightness,
                         t=t,
                         t_range=(t_min, t_max),
                         t_inv_fn=_invert)
        return


class GammaTransform(ColorTransform):
    def __init__(self, t=[1], t_min=0.667, t_max=1.5):
        super().__init__(fn=TVF.adjust_gamma,
                         t=t,
                         t_range=(t_min, t_max),
                         t_inv_fn=_invert)
        return


class SaturationTransform(ColorTransform):
    def __init__(self, t=[1], t_min=0.667, t_max=1.5):
        super().__init__(fn=TVF.adjust_saturation,
                         t=t,
                         t_range=(t_min, t_max),
                         t_inv_fn=_invert)
        return


class ContrastTransform(ColorTransform):
    def __init__(self, t=[1], t_min=0.667, t_max=1.5):
        super().__init__(fn=TVF.adjust_contrast,
                         t=t,
                         t_range=(t_min, t_max),
                         t_inv_fn=_invert)
        return


## --- SPATIAL TRANSFORMATIONS --- #


class SpatialTransform():
    def __init__(self, t=[1.0, 0.0, 0.0], optimize=True):
        self.t = np.array(t, dtype=np.float32)
        self.is_spatial = True
        self.optimize = optimize
        return

    def __call__(self, ims, t, invert=False):
        if invert:
            return self.invert(ims, t)
        return self.transform(ims, t)

    def get_opt_param(self):
        if self.optimize:
            return self.t
        return []

    def transform(self, ims, t):
        return self.affine_transform(ims, t)

    def invert(self, ims, t):
        return self.invert_affine_transform(ims, t)

    def affine_transform(self, ims, t):
        """
        Applies affine transformation to ims using parameters t. The variable t
        has 3 variables [scale, tx, ty], with identity being [0, 1, 1].
        Args:
            ims: tensor of size b x c x h xw
            t: transformation parameter b x 3
        Returns:
            transformed images
        """
        theta = torch.zeros_like(t).view(-1, 1, t.size(1)).repeat(1, 2, 1)
        theta[:, 0, 0] = t[:, 0]
        theta[:, 1, 1] = t[:, 0]
        theta[:, :, 2] = t[:, 1:]
        return F.grid_sample(ims, F.affine_grid(theta, ims.size()))

    def invert_affine_transform(self, ims, t):
        """
        Inverts the transformation applied by t.
        Args:
            ims: tensor of size b x c x h xw
            t: transformation parameter b x 3
        Returns:
            transformed images

        > t_ims = affine_transform(ims, t)
        > ims_hat = invert_affine_transform(t_ims, t)
        > # ims and and ims_hat should be close.
        """
        theta = torch.zeros_like(t).view(-1, 1, t.size(1)).repeat(1, 2, 1)
        theta[:, 0, 0] = 1.0 / t[:, 0]
        theta[:, 1, 1] = 1.0 / t[:, 0]
        theta[:, :, 2] = - (t[:, 1:] / t[:, :1])
        return F.grid_sample(ims, F.affine_grid(theta, ims.size()))

    def __str__(self):
        return 'SpatialTransform: {}'.format(super().__str__())
