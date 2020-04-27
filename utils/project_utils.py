from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import os, os.path as osp
import numpy as np
import torch
import torch.nn as nn
from im_utils import to_image, make_video, binarize, make_grid


def save_result(save_dir, fn, collages, target, weight, out, vars, losses,
                t_outs=None, t_out=None, t_target=None, transform=None):
    jpg_quality = [int(cv2.IMWRITE_JPEG_QUALITY), 100]

    idx = np.argmin(losses[-1][1]['vgg'])
    make_video(osp.join(save_dir, '{}.mp4'.format(fn)), collages, duration=5)
    cv2.imwrite(osp.join(save_dir, '{}.target.jpg'.format(fn)),
                to_image(target)[0], jpg_quality)
    cv2.imwrite(osp.join(save_dir, '{}.weight.jpg'.format(fn)),
                to_image(weight)[0], jpg_quality)

    if t_out is not None:
        cv2.imwrite(osp.join(save_dir, '{}.transform.final.jpg'.format(fn)),
                    to_image(out)[idx], jpg_quality)
        cv2.imwrite(osp.join(save_dir, '{}.final.jpg'.format(fn)),
                    to_image(t_out)[idx], jpg_quality)
    else:
        cv2.imwrite(osp.join(save_dir, '{}.final.jpg'.format(fn)),
                    to_image(out)[idx], jpg_quality)

    if t_target is not None:
        cv2.imwrite(osp.join(save_dir, '{}.transform.target.jpg'.format(fn)),
                    to_image(t_target)[0], jpg_quality)

    # Keep an uncompressed version of the target and weight
    np.save(osp.join(save_dir, '{}.loss.npy'.format(fn)), np.array(losses))

    if t_outs is not None:
        make_video(osp.join(save_dir, '{}.transform.mp4'.format(fn)),
                   t_outs[0], duration=5)
        make_video(osp.join(save_dir, '{}.transform.out.mp4'.format(fn)),
                   t_outs[1], duration=5)

    exp_vars = {'vars':vars, 'transform': transform}
    np.save(osp.join(save_dir, '{}.vars.npy'.format(fn)), exp_vars)
    return


def set_model_precision(model, precision='float'):
    # Make model half precision for faster run-time
    if precision == 'half':
        model.half()
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()
    elif precision == 'float':
        model.float()
    elif precision == 'double':
        model.double()
    return model


def prepare_variables(vars, precision='float'):
    if precision == 'float':
        return [v.float().cuda() for v in vars]
    elif precision == 'half':
        return [v.half().cuda() for v in vars]
    elif precision == 'double':
        return [v.double().cuda() for v in vars]
