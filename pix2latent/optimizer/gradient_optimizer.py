import time

import torch

from pix2latent.optimizer.base_optimizer import _BaseOptimizer
from pix2latent.utils.misc import progress_print
from pix2latent.utils.image import to_image, to_grid, binarize



class GradientOptimizer(_BaseOptimizer):
    """
    Basic gradient optimizer. Compatible with any gradient-based optimizer.
    This optimizer uses the optimizer defined in variable_manager
    """

    def __init__(self, *args, **kwargs):
        _BaseOptimizer.__init__(self, *args, **kwargs)
        return


    def optimize(self, num_samples, grad_steps, pbar=None):
        """
        Args
            num_samples (int): number of samples to optimize over
            grad_steps (int): number of gradient descent updates.
            pbar: progress bar such as tqdm or st.progress [Default: None]

        """
        self.losses, self.outs = [], []

        variables = self.var_manager.initialize(num_samples=num_samples)

        t_st = time.time()

        for i in range(grad_steps):
            self.step(variables, optimize=True, transform=(i == 0))

            if pbar is not None:
                pbar.progress(i / grad_steps)

            if self.log:
                if ((i + 1) % self.log_iter == 0) or (i + 1 == grad_steps):
                    self.log_result(variables, i + 1)

            if (i + 1) % self.show_iter == 0:
                t_avg = (time.time() - t_st) / self.show_iter
                progress_print('optimize', i + 1, grad_steps, 'c', t_avg)
                t_st = time.time()

        if self.log:
            return variables, self.outs, self.losses

        transform_out = to_grid(torch.stack(list(self.out.cpu().detach())))

        return variables, [transform_out], [[grad_steps, {'loss':self.loss}]]
