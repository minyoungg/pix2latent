from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.optim as optim

from . import distribution as dist
import pprint

from easydict import EasyDict as edict
import numpy as np


def split_vars(vars, size):
    """ Splits variable dictionary into mini chunks of dictionary """

    num_splits = int(np.ceil(vars.num_samples / float(size)))
    split_vars = []

    for i in range(num_splits):

        sub_vars = {}

        for var_type, var_dict in vars.items():
            if var_type in ['opt', 'num_samples']:
                continue

            sub_vars[var_type] = {}

            for var_name, var_data in var_dict.items():

                _data = var_data.data[i * size: (i + 1) * size]

                _num_samples = len(_data)

                sub_vars[var_type][var_name] = \
                        {'data': _data, 'hook_fn': var_data.hook_fn}

        sub_vars['opt'] = vars.opt
        sub_vars['num_samples'] = _num_samples

        split_vars.append(edict(sub_vars))

    return split_vars


def save_variables(save_path, variables):
    try:
        del vars['opt']
    except:
        pass

    for var_type, all_vars in variables.items():
        try:
            for var_name, var_data in all_vars.items():
                for i in range(len(variables[var_type][var_name].data)):
                    variables[var_type][var_name].data[i] = \
                        variables[var_type][var_name].data[i].detach().cpu()
        except:
            pass

    np.save(save_path, variables)
    return



class VariableManager():

    def __init__(self):
        """ A variable manager that creates variables for optimization """
        self.variable_info = {}
        return


    def __str__(self):
        """ Print format """
        fmt = '<Variable Manager>\n{}'
        return fmt.format(pprint.pformat(self.variable_info))


    def register(self,
                 variable_name,
                 shape,
                 var_type,
                 requires_grad=True,
                 default=None,
                 distribution=dist.TruncatedNormalModulo(sigma=1.0, trunc=2.0),
                 optimizer=optim.Adam,
                 learning_rate=0.05,
                 hook_fn=None,
                 grad_free=False,
                 ):
        """
        Registers a variable to the manager. The specs provided will be used
        for the variable intialization.

        Args:
            variable_name (str): the name of the variable
            shape (tuple): shape of the variable
            var_type: determines how the variables will be used by the
                optimizer. Must be one of [input, loss]
            requires_grad (bool): if True, the generated variables can be
                optimized. [Default: True]
            default (Tensor): default values of the model. [Default: None]
            distribution (dist): distribution to sample from. ignored if
                default is not None. [Default: None]
            optimizer (torch.optim): torch optimizer function. ignored if
                requires_grad is `False` [Default: optim.Adam]
            learning_rate (float): learning rate of the variable. ignored if
                requires_grad is `False`. [Default: 0.05]
            hook_fn (function): this function will run on this variable before
                optimizing. [Default: None]
            grad_free (bool, scalar, tuple): triggers the gradient-free
                optimizers to update the variable using methods like CMA.
                - if `grad_free` is `True`, it will be initialized using the
                normal distribution N(0, I),
                - if `grad_free` is a tuple, it assumes of the form (mu, sigma)
                and will be from the distribution N(mu, sigma), where mu is a
                list and sigma is a scalar. If mu or sigma is None, it will
                assume the default value. [Default: False]

        """

        if variable_name in self.variable_info:
            print('variable `{}`` already exists.'.format(variable_name))
            return False

        if default is not None:
            msg = 'default and shape must match but got {} vs {}'
            assert tuple(default.size()) == shape, \
                        msg.format(list(default.size()), shape)

        self.variable_info[variable_name] = {
            'shape': shape,
            'var_type': var_type,
            'requires_grad': requires_grad,
            'default': default,
            'distribution': distribution,
            'optimizer': optimizer,
            'learning_rate': learning_rate,
            'hook_fn': hook_fn,
            'grad_free': grad_free,
        }
        return True


    def unregister(self, *variable_names):
        """
        Removes a variable from the generation list.

        Args:
            variable_names (list or tuple): names of variables to remove

        """

        for v in variable_names:
            try:
                del self.variable_info[v]
            except:
                print('no variable named {}'.format(variable_name))

        return


    def edit_variable(self, variable_name, replace_dict):
        """
        Edits existing variable in the generation list

        Args:
            variable_name (str): name of the variable to replace.
            replace_dict (dict): dictionary consiting of variable attribute
                and the new value.

        Example:
            >>> var_man = VariableManger()
            >>> var_man.register('z', ...)
            >>> var_man.edit_variable('z', {'default': torch.zeros(1, 128)})
        """

        if variable_name not in self.variable_info.keys():
            print('variable `{}` does not exist'.format(variable_name))
            return False


        for k, v in replace_dict.items():
            if k not in self.variable_info[variable_name].keys():
                print('variable `{}` has no attribute {}'.format(k, v))
                return False

            self.variable_info[variable_name][k] = v

        return True

    @torch.no_grad()
    def initialize(self, num_samples):
        """
        Initializes variables from the generation list

        Args:
            num_samples (int): number of samples to generate for each variables

        """

        vars = {}
        params_to_optimize = []

        for v, spec in self.variable_info.items():

            if spec['default'] is not None:
                data = num_samples * [spec['default']]
            else:
                data = list(spec['distribution'](num_samples, spec['shape']))

            for i in range(len(data)):
                data[i] = data[i].detach().clone().cuda().requires_grad_(False)

            if spec['var_type'] not in vars.keys():
                vars[spec['var_type']] = {}

            vars[spec['var_type']][v] = \
                                {'data': data,
                                 'hook_fn': spec['hook_fn'],
                                 'grad_free': spec['grad_free'],
                                 'requires_grad': spec['requires_grad']}

            if not spec['requires_grad']:
                continue

            for d in data:
                params_to_optimize.append(
                        {'params': d.requires_grad_(True),
                         'lr': spec['learning_rate']}
                )


        vars['opt'] = spec['optimizer'](params_to_optimize)
        vars['num_samples'] = num_samples
        return edict(vars)
