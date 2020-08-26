import numpy as np

import torch

from pix2latent.edit.ganspace import biggan_components
from pix2latent.model import BigGAN



class BigGANLatentEditor():
    def __init__(self, model=None):
        if model is None:
            self.model = BigGAN().eval().cuda()
        return

    def load_result(self, var_path):
        """ load optimized result """
        self._var = np.load(var_path, allow_pickle=True).item()
        self._idx = np.argmin(self._var.loss[-1][1]['loss'])
        self._z = self._var.input.z.data[self._idx].unsqueeze(0).float().cuda()
        self._c = self._var.input.c.data[self._idx].unsqueeze(0).float().cuda()
        return

    def edit_class(self, cls_idx, alpha=1.0):
        """ edit class variable """
        c_orig = self._c
        c_edit = self.model.get_class_embedding(cls_idx)
        _c = (alpha * c_edit) + ((1.0 - alpha) * c_orig)

        with torch.no_grad():
            out = self.model(self._z, _c)[0]
        return out

    def edit_z(self, component, sigma):
        """ edit z-space using prinicipal component """
        if not hasattr(self, 'components'):
            self.components = biggan_components(self.model, self._c)

        u = self.components[component:component+1]

        with torch.no_grad():
            out = self.model(self._z + sigma * u, self._c)[0]
        return out


    def default(self):
        """ optimized result """
        with torch.no_grad():
            out = self.model(self._z, self._c)[0]
        return out





editor = BigGANLatentEditor()
editor.load_result('./results/biggan_256/cma_dog_182_n02093754_5756_lr_0.05_noise_0.05_trunc_2.0/vars.npy')
out = editor.edit_z(component=0, sigma=1.0)
out = editor.edit_class(cls_idx=220, alpha=1.0)
