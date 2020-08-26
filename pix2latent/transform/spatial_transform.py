import numpy as np

import torch
import torch.nn.functional as F

from pix2latent.transform.transform_utils import compute_pre_alignment
from pix2latent.transform.base_transform import TransformTemplate



class SpatialTransform(TransformTemplate):
    """
    Applies simple affine transformation to the image: scale and translation.
    This transformation assumes that the aspect ratio is fixed. There is also
    no sheering of the image but can be easily extended to model the full
    affine transformation matrix.
    """

    def __init__(self,
                 t=[1., 0., 0.],
                 identity_t=[1., 0., 0.],
                 pre_align=None,
                 sensitivity=0.1):
        """
        transformation parameter: [s_x, t_x, t_y]

        Args:
            identity_t (list): the identity transformation parameter. this is
                used as the center for the parameter search.
                [Default: [1., 0., 0.]]
            pre_align (image): if not None, uses a binary mask image to compute
                intial pre-alignment.
                [Default: [1., 0., 0.]]
            sensitivity (float): manually adjusted value for delta t.
                t = default_t + (sensitivity * delta_t) [Default: None]
        """

        self.identity_t = np.array(identity_t, dtype=np.float32)
        self.is_spatial = True
        self.sensitivity = sensitivity

        self.t = t
        if pre_align is not None:
            self.t = compute_pre_alignment(pre_align)

        self._t = torch.Tensor(self.t)
        return


    def __call__(self, ims, delta_t, invert=False):
        t = self._t.type_as(ims) + (self.sensitivity * delta_t)
        if invert:
            return self.invert_transform(ims, t)
        return self.transform(ims, t)


    def get_default_param(self, as_tensor=True):
        if as_tensor:
            return self._t
        return self.t


    def get_identity_param(self):
        if as_tensor:
            return torch.Tensor(self.identity_t)
        return self.identity_t


    def transform(self, ims, t):
        """
        Applies affine transformation to ims using parameters t. The variable t
        has 3 variables [scale, tx, ty], with identity being [0, 1, 1].
        Args:
            ims: tensor of size b x c x h x w
            t: transformation parameter b x 3
        Returns:
            transformed images
        """
        theta = torch.zeros_like(t).view(-1, 1, t.size(1)).repeat(1, 2, 1)
        theta[:, 0, 0] = t[:, 0]
        theta[:, 1, 1] = t[:, 0]
        theta[:, :, 2] = t[:, 1:]
        theta = theta.type_as(ims)
        return F.grid_sample(ims, F.affine_grid(theta, ims.size()))


    def invert_transform(self, ims, t):
        """
        Inverts the transformation applied by t.
        Args:
            ims: tensor of size b x c x h x w
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
