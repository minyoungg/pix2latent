from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
import torch
import numpy as np
import sys
import os
from torch import nn
import random


def set_seed(i):
    """ Does not set CMA seed. Might be a way to do it"""
    torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)
    return


def to_numpy(x):
    return x.detach().cpu().numpy()


def to_onehot(c):
    """ Creates a onehot torch tensor """
    onehot = torch.zeros((1, 1000))
    onehot[:, c] = 1.0
    return onehot


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


class HiddenPrints:
    """
    Suppress all print statements

    > with HiddenPrints():
    >     print('hi') # nothing happens
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class bcolors:
    HEADER = '\033[95m'
    b = blue = OKBLUE = '\033[94m'
    g = green = OKGREEN = '\033[92m'
    y = yellow = WARNING = '\033[93m'
    r = red = FAIL = '\033[91m'
    c = cyan = '\033[36m'
    lb = lightblue = '\033[94m'
    p = pink = '\033[95m'
    o = orange = '\033[33m'
    lc = lightcyan = '\033[96m'
    end = ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def color_str(string, color):
    """
    decorates string with terminal color code
    > s = color_str('hello', 'red') # s <- \033[91mhello\033[0m
    > print(s) # prints 'hello' in red
    """
    assert type(string)
    if not hasattr(bcolors, color):
        warnings.warn('Unknown color {}'.format(color))
        return string
    else:
        c = getattr(bcolors, color)
        e = getattr(bcolors, 'end')
        c_str = '{}{}{}'.format(c, string, e)
    return c_str


def cprint(print_str, color):
    """ its like print but with colors :) """
    c_str = color_str(print_str, color)
    print(c_str)
    return


def color_loss(loss):
    """ This function is specific to this project """
    c = 'red'
    if loss < 0.5:
        c = 'yellow'
    if loss < 0.1:
        c = 'green'
    if loss < 0.01:
        c = 'cyan'

    c = getattr(bcolors, c)
    e = getattr(bcolors, 'end')
    c_str = '{}{:.5f}{}'.format(c, loss, e)
    return c_str


def progress_print(phase, i, j, color='c'):
    p = color_str(phase, color)
    per = (100. * i) / j
    print('({}) progress {:.0f}% [{}/{}] '.format(p, per, i, j))
    return


if __name__ == '__main__':
    pbar = ProgressBar(50, 1)
    for _ in range(100):
        pbar()
