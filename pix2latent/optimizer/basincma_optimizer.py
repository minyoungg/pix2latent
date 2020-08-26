import time

from pix2latent.optimizer.base_optimizer import _BaseOptimizer
from pix2latent.optimizer.base_cma_optimizer import _BaseCMAOptimizer
from pix2latent.utils.misc import progress_print
from pix2latent.utils.image import to_image, to_grid, binarize

import torch



class BasinCMAOptimizer(_BaseOptimizer, _BaseCMAOptimizer):
    """
    CMA optimizer. Gradient descent can be used to further optimize the seeds
    from CMA.
    """

    def __init__(self, *args, **kwargs):
        _BaseOptimizer.__init__(self, *args, **kwargs)
        _BaseCMAOptimizer.__init__(self)
        return


    def optimize(self, meta_steps, grad_steps, last_grad_steps=300, pbar=None,
                 num_samples=None):
        """
        Args
            meta_steps (int): number of CMA updates
            grad_steps (int): number of gradient updates per CMA update.
            last_grad_steps (int): after the final iteration of BasinCMA
                further optimize the last drawn samples using gradient descent.
            pbar: progress bar such as tqdm or st.progress
            num_samples: must be None
        """

        assert num_samples == None, 'PyCMA optimizer has fixed sample size'

        self.setup_cma(self.var_manager)
        self.losses, self.outs, i = [], [], 0
        total_steps = meta_steps * grad_steps + last_grad_steps


        #####
        # -- BasinCMA optimization (outerloop CMA) -- #

        t_st = time.time()

        for meta_iter in range(meta_steps + 1):
            is_last_iter = (meta_iter == meta_steps)
            _grad_steps = last_grad_steps if is_last_iter else grad_steps

            variables = self.cma_init(self.var_manager)


            #####
            # -- Gradient optimization (innerloop SGD) -- #

            for j in range(_grad_steps):
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
                        progress_print(
                                'optimize', i + 1, total_steps, 'c', t_avg)
                        t_st = time.time()

            if not is_last_iter:
                self.cma_update(variables, inverted_loss=True)

        if self.log:
            return variables, self.outs, self.losses

        transform_out = to_grid(torch.stack(list(self.out.cpu().detach())))

        return variables, [transform_out], [[total_steps, {'loss':self.loss}]]
