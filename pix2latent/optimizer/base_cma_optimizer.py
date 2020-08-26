import torch
import cma
import numpy as np

from pix2latent.utils.misc import HiddenPrints, cprint
from pix2latent.utils.image import binarize


class _BaseCMAOptimizer():
    """
    Base template for CMA optimization. Should be used jointly with
    _BaseOptimizer.

    NOTE:
        Since the seeds are pre-determined by the CMA code-base, support
        to optimize multiple variables using CMA is disabled. NeverGrad
        optimizers should be able to support this feature.
    """

    def __init__(self):
        self.num_samples = -1
        self.cma_optimizers = {}
        self._sampled = {}
        return


    @torch.no_grad()
    def setup_cma(self, var_manager):
        """
        initializes CMA for variables that have the attribute `grad_free`

        Args
            var_manager (VariableManger): instance of the variable manager

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
                cma_opt = CMA(mu, sigma=1.0)


            self.cma_optimizers[(var_dict['var_type'], var_name)] = cma_opt

            self.num_samples = max(self.num_samples, cma_opt.batch_size())

        cprint('(cma-es) number of samples: {}'.format(self.num_samples), 'y')

        assert len(self.cma_optimizers.keys()) == 1, \
           'currently only a single input variable can be optimized via CMA '+\
           'but got: {}'.format(self.cma_optimizers.keys())
        return


    @torch.no_grad()
    def cma_init(self, var_manager):
        """
        initializes the provided variable from CMA

        Args:
            var_manager (VariableManger): instance of the variable manager
        """

        vars = var_manager.initialize(num_samples=self.num_samples)

        for (var_type, var_name), cma_opt in self.cma_optimizers.items():
            cma_data = cma_opt.ask()

            for i, d in enumerate(cma_data):
                vars[var_type][var_name].data[i].data = \
                            torch.Tensor(d).data.type_as(
                                vars[var_type][var_name].data[i].data)

            self._sampled[(var_type, var_name)] = cma_data

        return vars


    @torch.no_grad()
    def cma_update(self, variables, loss=None, inverted_loss=False):
        """
        Updates CMA distribution either with the provided loss or loss that
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

        for (var_type, var_name), cma_opt in self.cma_optimizers.items():

            cma_data = self._sampled[(var_type, var_name)]

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

            cma_opt.tell(cma_data, loss)
        return loss



class CMA():
    def __init__(self, mu=128 * [0], sigma=1.0, seed=None):
        """
        Wrapper class function for PyCMA. Since CMA does not allow for 1
        variable optimization, we will duplicate it to be 2 variables and
        only compute CMA on the first variable.

        Args:
            mu (list): a 1D array of CMA means. [Default: 128 * [0]]
            sigma (float): sigma for the CMA. This sigma is shared on all seeds
                [Default: 1.0]
            seed (int): seed used for reproducibility. [Default: None]

        Attribute:
            is_scalar (boolean): hack to go around the fact that CMA does not
                support scalar optimizer.
            cma (CMA): core CMA optimizer
        """

        options = {}

        if seed is not None:
            options['seed'] = seed
        self.is_scalar = False

        if len(mu) == 1:
            mu = list(mu) * 2
            options['CMA_on'] = 0
            self.is_scalar = True

        with HiddenPrints():
            self.cma = cma.CMAEvolutionStrategy(mu, sigma, options)
        return


    def batch_size(self):
        """ Returns the required batch size for CMA """
        return self.cma.sp.popsize


    def ask(self, batch_size=None):
        """ Asks for samples to evaluate. batch_size must be None to train """
        x = np.array(self.cma.ask(batch_size))

        if self.is_scalar:
            self._x = x
            self._x_proxy = x[:, :1]
            return self._x_proxy

        return x


    def tell(self, x, y):
        """ Apply CMA update """

        if self.is_scalar:
            assert x is self._x_proxy
            return self.cma.tell(self._x, y)

        return self.cma.tell(x, y)


    def mean(self):
        """ Returns the mean of the current CMA distribution """

        x = self.cma.mean

        if self.is_scalar:
            return x[:1]

        return x
