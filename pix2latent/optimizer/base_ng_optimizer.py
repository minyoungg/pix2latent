import nevergrad as ng

import numpy as np
import torch

from pix2latent.utils.image import binarize



class _BaseNevergradOptimizer():
    """
    Base template for NeverGrad optimization. Should be used jointly with
    BaseOptimizer.

    For full list of available optimizers
    > https://github.com/facebookresearch/nevergrad

    or ...
    > print(self.valid_methods)

    Args:
        method: nevergrad optimization method

    NOTE:
        nevergrad CMA have been observed to perform wrose than the original
        codebase. use with warning. nevergrad has a perk of being optimized
        in parallel, hence batch-size can be arbitrarily chosen.
    """

    def __init__(self, method):

        self.method = method
        self.valid_methods = [x[0] for x in ng.optimizers.registry.items()]

        # this is not an exhaustive list
        self.sequential_methods = ['SQPCMA', 'chainCMAPowell', 'Powell']
        self.is_sequential = self.method in self.sequential_methods

        if self.is_sequential:
            seq_msg = '{} is a sequential method. batch size is set to 1'
            cprint(seq_msg.format(self.method), 'y')

        assert self.method in self.valid_methods, \
                    f'unknown nevergrad method: {self.method}'

        self.ng_optimizers = {}
        self._sampled = {}
        return


    @torch.no_grad()
    def setup_ng(self, var_manager, budget):
        """
        initializes NeverGrad optimizer.

        Args
            var_manager (VariableManger): instance of the variable manager
            budget (int): number of optimization iteration.
        """

        for var_name, var_dict in var_manager.variable_info.items():

            if var_dict['grad_free'] is False:
                continue

            if type(var_dict['grad_free']) == tuple:
                mu, sigma = var_dict['grad_free']

                if mu is None:
                    mu = np.zeros(var_dict['shape'])

                if sigma is None:
                    sigma = 1.

                cma_opt = CMA(mu, sigma=sigma)

            else:
                mu = np.zeros(var_dict['shape'])
                sigma = 1.0

            opt_fn = ng.optimizers.registry[self.method]
            p = ng.p.Array(init=mu)#.set_mutation(sigma=sigma)
            ng_opt = opt_fn(parametrization=p, budget=budget)

            self.ng_optimizers[(var_dict['var_type'], var_name)] = ng_opt

        assert len(self.ng_optimizers.keys()) == 1, \
           'currently only a single input variable can be optimized via '+\
           'Nevergrad but got: {}'.format(self.ng_optimizers.keys())
        return


    @torch.no_grad()
    def ng_init(self, var_manager, num_samples):
        """
        Args
            var_manager (VariableManger): instance of the variable manager
            num_samples (int): number of samples for mini-batch optimization
        """
        if self.is_sequential:
            vars = var_manager.initialize(num_seeds=1)
            num_samples = 1
        else:
            vars = var_manager.initialize(num_samples=num_samples)

        for (var_type, var_name), ng_opt in self.ng_optimizers.items():
            ng_data = [ng_opt.ask() for _ in range(num_samples)]

            _ng_data = np.concatenate([x.args for x in ng_data])

            for i, d in enumerate(_ng_data):
                vars[var_type][var_name].data[i].data = \
                            torch.Tensor(d).data.type_as(
                                vars[var_type][var_name].data[i].data)

            self._sampled[(var_type, var_name)] = ng_data

        return vars


    @torch.no_grad()
    def ng_update(self, variables, loss=None, inverted_loss=False):

        """
        Updates NG distribution either with the provided loss or loss that
        is recomputed.

        Args:
            variables (dict): a dictionary instance generated from the
                variable manager.
            loss (array or list): a 1-dimensional array or list consisting of
                losses corresponding to each sample. If the loss is not
                provided, uses the variables to recompute the loss.
                [Default: None]
            inverted_loss (bool): if True, the loss is computed after inverting
                the generated images back to the original target. For example
                this is used to compute the loss on the original target.
                [Default: False]
        """

        for (var_type, var_name), ng_opt in self.ng_optimizers.items():

            ng_data = self._sampled[(var_type, var_name)]

            if loss is None:
                out, loss, _ = self.step(variables, optimize=False)

            if inverted_loss and hasattr(variables, 'transform'):

                target_type = \
                        self.var_manager.variable_info['target']['var_type']
                weight_type = \
                        self.var_manager.variable_info['weight']['var_type']

                target = self.var_manager.variable_info['target']['default']
                weight = self.var_manager.variable_info['weight']['default']

                target = target.unsqueeze(0).type_as(out)
                weight = weight.unsqueeze(0).type_as(out)

                t_fn = self.transform_fns['target']['fn']
                t_param = torch.stack(variables.transform.t.data)
                out = t_fn(out, t_param, invert=True)

                loss = self.loss_fn(out, target, binarize(weight))
                loss = loss.cpu().detach().numpy()

            for d, l in zip(ng_data, loss):
                ng_opt.tell(d, l)

        return
