from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cma
import numpy as np
from misc import HiddenPrints


class CMA():
    def __init__(self, mu=128 * [0], sigma=1.0, seed=None):
        """
        Wrapper class function for PyCMA. Since CMA does not allow for 1
        variable optimization, we will duplicate it to be 2 variables and
        only compute CMA on the first variable.

        Args:
            mu
                A 1D array of CMA means.
            sigma
                Sigma for the CMA. This sigma is shared on all CMA.
            seed
                Seed of the CMA
        """
        options = {}
        if seed is not None:
            options['seed'] = seed
        self.is_1d = False
        if len(mu) == 1:
            mu = list(mu) * 2
            options['CMA_on'] = 0
            self.is_1d = True
        with HiddenPrints():
            self.cma = cma.CMAEvolutionStrategy(mu, sigma, options)
        return

    def batch_size(self):
        """ Returns the required batch size for CMA """
        return self.cma.sp.popsize

    def ask(self, batch_size=None):
        """ Asks for samples to evaluate. batch_size must be None to train """
        x = np.array(self.cma.ask(batch_size))
        if self.is_1d:
            self._x = x
            self._x_proxy = x[:, :1]
            return self._x_proxy
        return x

    def tell(self, x, y):
        """ Apply CMA update """
        if self.is_1d:
            assert x is self._x_proxy
            return self.cma.tell(self._x, y)
        return self.cma.tell(x, y)

    def mean(self):
        """ Returns the mean of the current CMA distribution """
        x = self.cma.mean
        if self.is_1d:
            return x[:1]
        return x
