from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch
import torch.nn as nn

import pytorch_pretrained_biggan as ppb
from pix2latent.utils.misc import \
        HiddenPrints, replace_to_inplace_relu, remove_spectral_norm


class BigGAN(nn.Module):
    """
    Wrapper class for HuggingFaces BigGAN. Nothing new here, simplifies
    optimizing for continuous class vector.

    DataParallelizing this version runs slow.
    """

    def __init__(self, model_version='biggan-deep-256'):
        super(BigGAN, self).__init__()
        with HiddenPrints():
            biggan = ppb.BigGAN.from_pretrained(model_version)
            self.embeddings = biggan.embeddings.eval()
            self.generator = biggan.generator.eval()

            replace_to_inplace_relu(self.generator)

            remove_spectral_norm(self.generator)

        return


    def get_class_embedding(self, cls):
        with torch.no_grad():
            if type(cls) == int:
                # converts to one hot
                c = torch.zeros(1, 1000).float().cuda()
                c[:, cls] = 1
            elif len(cls.size()) == 2:
                c = cls
            else:
                raise ValueError
            return self.embeddings(c)


    def forward(self, z=None, c=None, truncation=1.0):
        assert 0 < truncation <= 1

        assert len(z.size()) == 2, 'expected z to be 2D'
        assert len(c.size()) == 2, 'expected c to be 2D'
        assert c.size(1) == 128, \
                'expected c to have dim (?, 128) but got {}'.format(c.size())

        return self.generator(torch.cat((z, c), dim=1), truncation)
