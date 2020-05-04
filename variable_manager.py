from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.optim as optim

from easydict import EasyDict as edict
import numpy as np
import checks


def str_to_dtype(precision):
    if precision == 'half':
        dtype = torch.float16
    elif precision == 'float':
        dtype = torch.float32
    elif precision == 'double':
        dtype = torch.float64
    else:
        raise NotImplementedError('Does not support: {}'.format(precision))
    return dtype


def get_opt_fn(opt_name):
    """
    Returns gradient optimizer by name. Add to list if needed. Any optimizer
    added should take in 2 variables: [params, lr]. Use lambda function if
    this is a problem.
    """

    if opt_name == 'adam':
        return optim.Adam

    elif opt_name == 'lbfgs':
        # Consider using community version. Omitted for public release.
        # https://github.com/hjmshi/PyTorch-LBFGS
        # http://sagecal.sourceforge.net/pytorch/index.html
        return lambda params, lr, eps: \
            optim.LBFGS(params, lr=lr, line_search_fn='strong_wolfe')

    raise NotImplementedError('Does not support: {}'.format(opt_name))


class VariableManager():
    def __init__(self, opt='adam', optimize_z=True, lr=0.05, cv_lr=0.005,
                 optimize_t=False, t_lr=0.05, precision='float',
                 cv_search_method='none'):
        """ A variable manager that creates variables for optimization """
        self.opt_fn = get_opt_fn(opt)
        self.lr, self.cv_lr, self.t_lr = lr, cv_lr, t_lr
        self.optimize_z = optimize_z
        self.cv_search_method = cv_search_method
        self.optimize_t = optimize_t
        self.precision = precision
        self.num_seeds = 1
        self.dtype = str_to_dtype(self.precision)
        self.optimize_cv = self.cv_search_method != 'none'
        return

    def vars_to_param(self, variables):
        """ Convert variable dict to torch optimizer friendly format  """
        opt_params = []
        for k, v in variables.items():
            if v is None:
                continue

            if k == 'opts' or not v.requires_grad:
                continue

            for i in range(len(v['data'])):
                opt_params.append({'params': v['data'][i].requires_grad_(),
                                   'lr': v['lr']})
        return opt_params

    def set_default(self, num_seeds=None, z=None, cv=None, t=None, target=None,
                    weight=None):
        """
        Sets default parameters so you can call init() without repeatedly
        providing these arguments. All arguments should have a batch-size 1
        """
        if z is not None:
            if type(z) in [list, tuple]:
                assert checks.is_single_1d(z[0])
                self.z, self.z_sigma = z[0].clone(), z[1]
            else:
                assert checks.is_single_1d(z)
                self.z, self.z_sigma = z.clone(), 1.0

        if cv is not None:
            assert checks.is_single_1d(cv) or type(cv) is tuple
            self.cv = cv.clone()

        if t is not None:
            assert checks.is_single_1d(t)
            self.t = t.clone()

        if target is not None:
            checks.check_input(target)
            self.target = target.clone()

        if weight is not None:
            checks.check_input(weight)
            self.weight = weight.clone()

        if num_seeds is not None:
            assert num_seeds > 0
            self.num_seeds = num_seeds
        return

    def init(self, num_seeds=None, target=None, weight=None, z=None, cv=None,
             t=None, precision='float'):
        """
        Initializes necessary variables. Creates a variable object.
        Creates num_seeds number of variables. The variables do not share the
        momentum statistics. Varible object is an easydict object:

        Args:
            num_seeds:
                Number of random seeds to generate.
            target:
                The target image. Must be of shape 1 x 3 x H x W
            weight:
                The weight image should be the same dimension as target
            z:
                Latent variable. If not provided, samples from truncated
                torch.randn()
            cv:
                Continuous class vector and NOT a one hot encoding. If not
                provided initializes from torch.randn().
            t:
                Transformation parameter. If not provided, it is set to None.
            precision:
                choose from [double, float, half]

        Returns
            Creates variable object that is used by optimizer

        Note
            A variable object is a dictionary of individual variable. Each
            variable has 3 attributes, (data, opt, requires_grad).

            data: list of tensors
            opt: optimized if True
            requires_grad: optimized by the optimizer if True

            >> v = variable_manager.init()
            >> v.z.data  # [Tensor, .., Tensor]
            >> v.opt # Optimizer
        """

        ## --- use preset if not provided --- ##

        if hasattr(self, 'num_seeds') and num_seeds is None:
            num_seeds = self.num_seeds

        if hasattr(self, 'z') and z is None:
            z = self.z.clone() + \
                self.z_sigma * sample_truncated_normal(num_seeds, 2.0)
        else:
            if z is None:
                z = sample_truncated_normal(num_seeds, 2.0)
            else:
                z = z.detach().clone()

        if hasattr(self, 'cv') and cv is None:
            cv = self.cv.clone()
        else:
            if cv is None:
                cv = sample_truncated_normal(num_seeds, 2.0)
            else:
                cv = cv.detach().clone()

        if hasattr(self, 't') and t is None:
            t = self.t.clone()

        if hasattr(self, 'target'):
            if target is None:
                target = self.target.clone()
            else:
                raise RuntimeError('target must be provided or set as default')

        if hasattr(self, 'weight') and weight is None:
            weight = self.weight.clone()

        checks.check_input(target)

        if weight is not None:
            checks.check_input(weight)

        if cv.size(0) == 1:
            cv.data = cv.repeat(num_seeds, 1)

        # --- cast to desired precision --- #
        target, weight, z, cv, t, eps = self.cast(target, weight, z, cv, t)

        # --- construct variable object --- #
        variables = edict()
        variables.z = edict({'data': list(z),
                             'lr': self.lr,
                             'requires_grad': self.optimize_z})

        variables.cv = edict({'data': list(cv),
                              'lr': self.cv_lr,
                              'requires_grad': self.optimize_cv})

        variables.target = edict({'data': list(target),
                                  'lr': None,
                                  'requires_grad': None})

        variables.weight = edict({'data': list(weight),
                                  'lr': None,
                                  'requires_grad': None})

        if t is not None or self.optimize_t:
            if checks.is_single_1d(t):
                t = t.detach().clone().repeat(num_seeds, 1)

            t = t.type(self.dtype)
            variables.t = edict({'data': list(t),
                                 'lr': self.t_lr,
                                 'requires_grad': self.optimize_t})

        opt_params = self.vars_to_param(variables)
        variables.opt = self.opt_fn(opt_params, lr=self.lr, eps=eps)
        variables.num_seeds = num_seeds
        return variables

    def cast(self, *args):
        """ Casts all argument to specified precision. """
        casted = [arg if arg is None else arg.type(self.dtype).cuda()
                  for arg in args]

        # For numerical stability
        if self.precision == 'half':
            casted.append(1e-4)
        elif self.precision == 'float':
            casted.append(1e-8)
        elif self.precision == 'double':
            casted.append(1e-16)
        return casted


def sample_truncated_normal(num_seeds, truncate=2.0):
    """ Truncate the values using fmod. fmod applies mod to floating point """
    z = torch.randn((num_seeds, 128)).float().cuda()
    if truncate is not False or truncate is not None:
        z = torch.fmod(z, truncate)
    return z


def override_variables(variables, override_variables):
    """
    Overides existing variables in place.
    args:
        var: an easy dict with all the variables
        override_var: a list consisting of [var_str, variable_data]
                      variable_data can be either torch or numpy tensor.
                      if numpy, it will automatically converted.
    """
    e_msg = 'expected variable {} to be of size {} but got {}'
    for k, v in override_variables:
        for i in range(len(v)):
            assert len(variables[k].data) == len(v), \
                '{}: {} vs {}'.format(k, np.shape(variables[k].data), np.shape(v))

            try:  # hacky way to check if override vars is numpy or pytorch.
                _v = torch.from_numpy(v[i])
            except:
                _v = v[i]

            var_size = _v.size()
            new_var_size = variables[k].data[i].data.size()

            assert var_size == new_var_size, \
                e_msg.format(k, var_size, new_var_size)

            dtype = variables[k].data[i].data.dtype
            variables[k].data[i].data = _v.cuda().to(dtype)
    return


def split_vars(vars, size):
    """ Splits variable dictionary into mini chunks of dictionary """
    num_splits = int(np.ceil(len(vars.z.data) / float(size)))
    split_vars = []
    ignore = ['opt', 'num_seeds']
    for i in range(num_splits):
        sub_vars = edict({k: {'data': v.data[i * size: (i + 1) * size],
                              'lr': v.lr,
                              'requires_grad': v.requires_grad}
                          for k, v in vars.items()
                          if v is not None and k not in ignore})
        sub_vars.opt = vars.opt
        split_vars.append(sub_vars)
    return split_vars
