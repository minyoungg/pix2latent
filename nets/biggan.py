from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch
import torch.nn as nn

import pytorch_pretrained_biggan as ppb
from misc import HiddenPrints


class BigGAN(nn.Module):
    """
    Wrapper class for HuggingFaces BigGAN. Nothing new here, simplifies
    optimizing for continuous class vector.

    DataParallelizing this version runs slow.
    """

    def __init__(self):
        super(BigGAN, self).__init__()
        with HiddenPrints():
            biggan = ppb.BigGAN.from_pretrained('biggan-deep-256')
            self.embeddings = biggan.embeddings
            self.generator = biggan.generator
        return

    def forward(self, z=None, c=None, truncation=1.0, embed_class=False):
        assert 0 < truncation <= 1

        if embed_class:
            assert c is not None and z is None
            return self.embeddings(c)

        assert z.size(1) == c.size(1), 'class embedding must be continuous.'
        return self.generator(torch.cat((z, c), dim=1), truncation)


if __name__ == '__main__':
    model = nn.DataParallel(BigGAN().cuda())
    c = torch.ones(4, 1000).float().cuda()
    z = torch.ones(4, 128).float().cuda()

    cv = model(c=c, embed_class=True)
    im = model(z=z, c=cv)
