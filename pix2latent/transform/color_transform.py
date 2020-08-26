from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torchvision.transforms.functional as TVF



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


#NOTE: Since lambda functions cant be pickled easily.
def _negate(x):
    return -x


def _invert(x):
    return 1.0 / x
