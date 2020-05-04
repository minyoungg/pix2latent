from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import models
import torch
from torch import nn
import checks
from misc import HiddenPrints


def l1_loss(out, target):
    """ computes loss = | x - y |"""
    return torch.abs(target - out)


def l2_loss(out, target):
    """ computes loss = (x - y)^2 """
    return ((target - out) ** 2)


def invertibility_loss(ims, target_transform, transform_params, mask=None):
    """ Computes invertibility loss MSE(ims - T^{-1}(T(ims))) """
    if ims.size(0) == 1:
        ims = ims.repeat(len(transform_params), 1, 1, 1)
    transformed = target_transform(ims, transform_params)
    inverted = target_transform(transformed, transform_params, invert=True)
    if mask is None:
        return torch.mean((ims - inverted) ** 2, [1, 2, 3])
    return masked_l2_loss(ims, inverted, mask)


def masked_l1_loss(out, target, mask):
    checks.check_loss_input(out, target, mask)
    if mask.size(0) == 1:
        mask = mask.repeat(out.size(0), 1, 1, 1)
    if target.size(0) == 1:
        target = target.repeat(out.size(0), 1, 1, 1)

    loss = l1_loss(out, target)
    n = torch.sum(loss * mask, [1, 2, 3])
    d = torch.sum(mask, [1, 2, 3])
    return (n / d)


def masked_l2_loss(out, target, mask):
    checks.check_loss_input(out, target, mask)
    if mask.size(0) == 1:
        mask = mask.repeat(out.size(0), 1, 1, 1)
    if target.size(0) == 1:
        target = target.repeat(out.size(0), 1, 1, 1)
    loss = l2_loss(out, target)
    n = torch.sum(loss * mask, [1, 2, 3])
    d = torch.sum(mask, [1, 2, 3])
    return (n / d)


class ReconstructionLoss(nn.Module):
    """ Reconstruction loss with spatial weighting """

    def __init__(self, loss_type='l1'):
        super(ReconstructionLoss, self).__init__()
        if loss_type in ['l1', 1]:
            self.loss_fn = l1_loss
        elif loss_type in ['l2', 2]:
            self.loss_fn = l2_loss
        else:
            raise ValueError('Unknown loss_type {}'.format(loss_type))
        return

    def __call__(self, im0, im1, w=None):
        checks.check_loss_input(im0, im1, w)
        loss = self.loss_fn(im0, im1)
        if w is not None:
            n = torch.sum(loss * w, [1, 2, 3])
            d = torch.sum(w, [1, 2, 3])
            loss = n / d
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self, net='vgg', use_gpu=True, precision='float'):
        """ LPIPS loss with spatial weighting """
        super(PerceptualLoss, self).__init__()
        with HiddenPrints():
            self.lpips = models.PerceptualLoss(model='net-lin',
                                               net=net,
                                               spatial=True,
                                               use_gpu=use_gpu)
        if use_gpu:
            self.lpips = nn.DataParallel(self.lpips).cuda()
        if precision == 'half':
            self.lpips.half()
        elif precision == 'float':
            self.lpips.float()
        elif precision == 'double':
            self.lpips.double()
        return

    def forward(self, im0, im1, w=None):
        """ ims have dimension BCHW while mask is 1HW """
        checks.check_loss_input(im0, im1, w)
        # lpips takes the sum of each spatial map
        loss = self.lpips(im0, im1)
        if w is not None:
            n = torch.sum(loss * w, [1, 2, 3])
            d = torch.sum(w, [1, 2, 3])
            loss = n / d
        return loss

    def __call__(self, im0, im1, w=None):
        return self.forward(im0, im1, w)
