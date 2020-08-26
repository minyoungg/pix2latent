import time

import numpy as np
import nevergrad as ng
import torch

from pix2latent.optimizer.base_optimizer import _BaseOptimizer
from pix2latent.optimizer.base_ng_optimizer import _BaseNevergradOptimizer
from pix2latent.utils.misc import progress_print
from pix2latent.utils.image import to_image, to_grid, binarize



class NevergradOptimizer(_BaseOptimizer, _BaseNevergradOptimizer):

    def __init__(self, method, *args, **kwargs):
        _BaseOptimizer.__init__(self, *args, **kwargs)
        _BaseNevergradOptimizer.__init__(self, method=method)
        return


    def optimize(self, num_samples, meta_steps, grad_steps=0, pbar=None):
        """
        Args
            num_samples (int): number of samples to optimize
            grad_steps (int): number of gradient descent updates.
            meta_steps (int): number of Nevergrad updates
            grad_steps (int): number of gradient updates to apply after
                Nevergrad optimization. [Default: 0]
            pbar:
                progress bar such as tqdm or st.progress
        """

        self.setup_ng(self.var_manager, meta_steps) # double check if budget is number of times you call or call sequentially
        self.losses, self.outs, i = [], [], 0
        total_steps = meta_steps + grad_steps


        #####
        # -- Nevergrad optimization (no gradient descent) -- #
        t_st = time.time()

        for _ in range(meta_steps):
            variables = self.ng_init(self.var_manager, num_samples)

            self.step(variables, optimize=False, transform=False)
            i += 1

            if self.log:
                if (i % self.log_iter == 0) or (i == grad_steps):
                    self.log_result(variables, i)

            self.ng_update(variables, inverted_loss=True)

            if pbar is not None:
                pbar.progress(i / total_steps)
            else:
                if i % self.show_iter == 0:
                    t_avg = (time.time() - t_st) / self.show_iter
                    progress_print('optimize', i, total_steps, 'c', t_avg)
                    t_st = time.time()


        #####
        # -- Finetune Nevergrad result with ADAM optimization -- #

        variables = self.ng_init(self.var_manager, num_samples)

        for j in range(grad_steps):
            self.step(variables, optimize=True, transform=(j == 0))
            i += 1

            if self.log:
                if ((i + 1) % self.log_iter == 0) or (i + 1 == grad_steps):
                    self.log_result(variables, i + 1)

            if pbar is not None:
                pbar.progress(i / total_steps)
            else:
                if (i + 1) % self.show_iter == 0:
                    t_avg = (time.time() - t_st) / self.show_iter
                    progress_print('optimize', i + 1, total_steps, 'c', t_avg)
                    t_st = time.time()


        if self.log:
            return variables, self.outs, self.losses

        transform_out = to_grid(torch.stack(list(self.out.cpu().detach())))

        return variables, [transform_out], [[total_steps, {'loss':self.loss}]]
