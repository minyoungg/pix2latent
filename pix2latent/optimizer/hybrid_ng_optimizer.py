import time

from pix2latent.optimizer.base_optimizer import _BaseOptimizer
from pix2latent.optimizer.base_ng_optimizer import _BaseNevergradOptimizer
from pix2latent.utils.misc import progress_print
from pix2latent.utils.image import to_image, to_grid, binarize

import torch



class HybridNevergradOptimizer(_BaseOptimizer, _BaseNevergradOptimizer):
    """
    Hybrid Nevergrad optimizer.
    """

    def __init__(self, method, *args, **kwargs):
        _BaseOptimizer.__init__(self, *args, **kwargs)
        _BaseNevergradOptimizer.__init__(self, method=method)
        return


    def optimize(self, num_samples, meta_steps, grad_steps, last_grad_steps=300,
                 pbar=None):
        """
        Args
            num_samples (int): number of samples to optimize
            meta_steps (int): number of Nevergrad updates
            grad_steps (int): number of gradient updates per Nevergrad update.
            last_grad_steps (int): after the final iteration of hybrid
                optimization further optimize the last drawn samples using
                gradient descent.
            pbar: progress bar such as tqdm or st.progress
        """

        self.losses, self.outs, i = [], [], 0
        total_steps = meta_steps * grad_steps + last_grad_steps
        self.setup_ng(self.var_manager, budget=meta_steps * grad_steps)


        #####
        # -- Hybrid optimization (outerloop Nevergrad) -- #

        t_st = time.time()

        for meta_iter in range(meta_steps + 1):
            is_last_iter = (meta_iter == meta_steps)
            _grad_steps = last_grad_steps if is_last_iter else grad_steps

            variables = self.ng_init(self.var_manager, num_samples)


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
                self.ng_update(variables, inverted_loss=True)

        if self.log:
            return variables, self.outs, self.losses

        transform_out = to_grid(torch.stack(list(self.out.cpu().detach())))

        return variables, [transform_out], [[total_steps, {'loss':self.loss}]]
