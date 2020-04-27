from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch

## -- Assertion checks -- ##

def check_input(x):
    oob_msg = 'image is outside the range [-1, 1] but got {} {}'
    assert x.min() >= -1.0 or x.max() <= 1.0, oob.format(x.min(), x.max())
    assert x.size(0) == 1, 'only supports batch size 1 {}'.format(x.size())
    assert len(list(x.size())) == 4
    return


def check_loss_input(im0, im1, w):
    """ im0 is out and im1 is target and w is mask"""
    e_msg = 'Dim 0 doesnt match {} vs {}'.format(w.size(0), im0.size(0))

    if w is not None:
        assert w.size(0) == im0.size(0) or w.size(1) == 1, e_msg

    assert im0.size() == im1.size(), '{} vs {}'.format(im0.size(), im1.size())
    return


## -- Boolean statements -- ##

def is_single_1d(z):
    """ checks whether the vector has the form [1 x N] """
    if type(z) is not torch.Tensor:
        return False

    if z.size(0) == 1 and len(z.size()) == 2:
        return True
    return False
