from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import os.path as osp
import sys
import logging
import traceback

import torch
from torch import nn
from pix2latent.utils.misc import HiddenPrints


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
    if mask.size(0) == 1:
        mask = mask.repeat(out.size(0), 1, 1, 1)
    if target.size(0) == 1:
        target = target.repeat(out.size(0), 1, 1, 1)

    loss = l1_loss(out, target)
    n = torch.sum(loss * mask, [1, 2, 3])
    d = torch.sum(mask, [1, 2, 3])
    return (n / d)


def masked_l2_loss(out, target, mask):
    if mask.size(0) == 1:
        mask = mask.repeat(out.size(0), 1, 1, 1)
    if target.size(0) == 1:
        target = target.repeat(out.size(0), 1, 1, 1)
    loss = l2_loss(out, target)
    n = torch.sum(loss * mask, [1, 2, 3])
    d = torch.sum(mask, [1, 2, 3])
    return (n / d)


def weight_regularization(orig_model, curr_model, reg='l1', weight_dict=None):
    w = 1.0
    reg_loss = 0.0
    orig_state_dict = orig_model.state_dict()
    for param_name, curr_param in curr_model.named_parameters():
        if 'bn' in param_name:
            continue
        orig_param = orig_state_dict[param_name]

        if reg == 'l1':
            l = torch.abs(curr_param - orig_param).mean()
        elif reg == 'l2':
            l = ((curr_param - orig_param) ** 2).mean()
        elif reg == 'inf':
            l = torch.max(torch.abs(curr_param - orig_param))

        if weight_dict is not None:
            w = weight_dict[param_name]
        reg_loss += w * l
    return reg_loss


class ProjectionLoss(nn.Module):
    """ The default loss that is used in the paper """

    def __init__(self, lpips_net='alex', beta=10):
        super().__init__()
        self.beta = beta
        self.rloss_fn = ReconstructionLoss()
        self.ploss_fn = PerceptualLoss(net=lpips_net)
        return


    def __call__(self, output, target, weight=None, loss_mask=None):
        rec_loss = self.rloss_fn(output, target, weight, loss_mask)
        per_loss = self.ploss_fn(output, target, weight, loss_mask)
        return rec_loss + (self.beta * per_loss)



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

    def __call__(self, output, target, weight=None, loss_mask=None):
        loss = self.loss_fn(output, target)
        if weight is not None:
            _weight = weight if loss_mask == None else (loss_mask * weight)
            n = torch.sum(loss * _weight, [1, 2, 3])
            d = torch.sum(_weight, [1, 2, 3])
            loss = n / d
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self, net='vgg', use_gpu=True, lpips_dir=None):
        """ LPIPS loss with spatial weighting """
        super(PerceptualLoss, self).__init__()
        lpips_dir = self.download(lpips_dir=lpips_dir)

        # consider removing from path  after importing
        sys.path.insert(1, lpips_dir)

        with HiddenPrints():
            import models
            self.lpips = models.PerceptualLoss(model='net-lin',
                                               net=net,
                                               spatial=True,
                                               use_gpu=use_gpu)

        self.lpips = nn.DataParallel(self.lpips).cuda()
        return

    def __call__(self, output, target, weight=None, loss_mask=None):
        """ ims have dimension BCHW while mask is 1HW """
        # lpips takes the sum of each spatial map
        loss = self.lpips(output, target)
        if weight is not None:
            _weight = weight if loss_mask == None else (loss_mask * weight)
            n = torch.sum(loss * _weight, [1, 2, 3])
            d = torch.sum(_weight, [1, 2, 3])
            loss = n / d
        return loss

    def download(self, lpips_dir=None):
        """ https://github.com/richzhang/PerceptualSimilarity """
        from git import Repo
        dir_path = os.path.dirname(os.path.realpath(__file__))
        git_url = "https://github.com/richzhang/PerceptualSimilarity"

        if lpips_dir is None:
            lpips_dir = osp.join(dir_path, 'PerceptualSimilarity')

        if not osp.exists(lpips_dir):
            print('Could not find LPIPS .. cloning from git')

            try:
                print(f'cloning `{git_url}` to `{lpips_dir}`')
                Repo.clone_from(git_url, lpips_dir)

            except Exception as e:
                logging.error(traceback.format_exc())
                print('Failed to clone the repo. Download it manually ' + \
                      'and set the path with `lpips_dir` argument.')

        return lpips_dir
