import time
import torch
import numpy as np
import cv2

from pix2latent.optimizer.base_optimizer import _BaseOptimizer
from pix2latent.optimizer.base_cma_optimizer import _BaseCMAOptimizer
from pix2latent.utils.misc import progress_print
from pix2latent.utils.image import to_image, to_grid, binarize


"""
TODO
Simplify integration with the variable manage.
Define inner loop variable and outer loop variable.
Determine latent propagation per variable.
"""


class TransformBasinCMAOptimizer(_BaseOptimizer, _BaseCMAOptimizer):
    """
    transformation search using basin-cma like optimization.
    variable propagation will use the best performing seed from the previous
    cma update to initialize new seeds. use variable propagation on the latent
    variable that you are optimizing over to significantly reduce run-time.
    """
    def __init__(self, *args, **kwargs):
        _BaseOptimizer.__init__(self, *args, **kwargs)
        _BaseCMAOptimizer.__init__(self)
        self.variables_to_propagate = []
        return


    @torch.no_grad()
    def vis_transform(self, variables):
        target = torch.stack(variables.output.target.data)
        weight = torch.stack(variables.output.weight.data)

        transform_im = to_image(to_grid(target * weight), cv2_format=False)

        if self.log_resize_factor is not None:
            transform_im = cv2.resize(
                                np.array(transform_im, dtype=np.uint8), None,
                                fx=self.log_resize_factor,
                                fy=self.log_resize_factor,
                                interpolation=cv2.INTER_AREA,
                            )

        self.transform_outs.append(transform_im)
        return


    def set_variable_propagation(self, variable_name):
        """ tells optimizer which variable to propagate """
        if variable_name in self.variables_to_propagate:
            print(f'variable {variable_name} already exists')
            return

        self.variables_to_propagate.append(variable_name)
        return


    def del_variable_propagation(self, variable_name):
        """ deletes variable that is tracked """
        if variable_name in self.variables_to_propagate:
            print(f'variable {variable_name} already exists')
            return

        self.variables_to_propagate.remove(variable_name)
        return


    @torch.no_grad()
    def update_propagation_variable_statistic(self, variables, ema_beta=0.5):
        """
        keeps a moving average of the optimize input variables. maybe better
        way is to move towards the latent code if it has better loss

        Args:
            variables (dict): variables from VariableManager
            ema_beta (float): moving average decay rate. 1 forgets everything
                and 0 remember nothing.

        """

        for var_name in self.variables_to_propagate:

            if var_name not in variables.input:
                msg = f'variable propagation is set for {var_name} but ' + \
                      'no such variable was found'
                raise RunTimeError(msg)

            var_data = variables.input[var_name]

            if var_name not in self.vp_means.keys():
                self.vp_means[var_name] = \
                            torch.stack(var_data.data).mean(0)
                #self.vp_means[var_name] = torch.zeros_like(var_data.data[0])

            # move in the direction of the seed that performed the best
            current_mean = var_data.data[np.argmin(self.loss)]
            #current_mean = torch.stack(var_data.data).mean(0)

            self.vp_means[var_name] = \
                    ((1.0 - ema_beta) * self.vp_means[var_name]) + \
                    (ema_beta * current_mean)

        return


    @torch.no_grad()
    def propagate_variable(self, variables, curr_iter, total_iter,
                           magnitude=1.0, renormalize=True):
        """
        resamples input variables from the moving average. if first time being
        tracked, it will be initialized with the mean of the variables. the
        noise added to the variables are proportionate to the progress

        Args:
            variables (dict): variables from VariableManager
            curr_iter (int): current iteration
            total_iter (int): total training iterations
            magnitude (float): noise magnitude
            renormalize (bool): standardizes the latent variables to be N(0,I)
                this helps in cases where latent variables collapse into
                a degenerate solution.
        """

        for var_name in self.variables_to_propagate:

            if var_name not in variables.input:
                msg = f'variable propagation is set for {var_name} but ' + \
                      'no such variable was found'
                raise RunTimeError(msg)

            var_data = variables.input[var_name]

            # initialize the running mean
            if var_name not in self.vp_means.keys():
                self.vp_means[var_name] = torch.stack(var_data.data).mean(0)
                #self.vp_means[var_name] = torch.zeros_like(var_data.data[0])


            # amount of noise to add to the running mean
            z_sigma = magnitude * (1 - (curr_iter / float(total_iter)))


            # resample using running mean
            for i in range(len(var_data.data)):
                _data = (self.vp_means[var_name] + \
                        (z_sigma * torch.randn_like(var_data.data[i]))).data

                if renormalize:
                    _data = (_data - _data.mean()) / _data.std()

                var_data.data[i].data = _data

        return


    def get_candidate(self):
        return self._candidate


    def optimize(self, meta_steps, grad_steps, last_grad_steps=None, pbar=None):
        """
        Args
            meta_steps (int): number of CMA updates
            grad_steps (int): number of gradient updates per CMA update.
            pbar: progress bar such as tqdm or st.progress

        """

        self.setup_cma(self.var_manager)
        self.losses, self.outs, self.transform_outs, i = [], [], [], 0
        self._best_loss, self._candidate = 999, None
        self.vp_means = {}
        self.transform_tracked = []

        if last_grad_steps is None:
            last_grad_steps = grad_steps

        total_steps = (meta_steps - 1) * grad_steps + last_grad_steps


        #####
        # -- BasinCMA optimization (outerloop CMA) -- #

        t_st = time.time()

        for meta_iter in range(meta_steps):
            is_last_iter = (meta_iter + 1 == meta_steps)
            _grad_steps = last_grad_steps if is_last_iter else grad_steps

            variables = self.cma_init(self.var_manager)

            if meta_iter > 0:
                self.propagate_variable(variables, meta_iter, meta_steps)

            self.transform_tracked.append(
                torch.stack(variables.transform.t.data).cpu().detach().clone()
            )

            #####
            # -- Gradient optimization (innerloop SGD) -- #

            for j in range(_grad_steps):

                self.step(variables, optimize=True, transform=(j == 0))
                i += 1

                if self.log and (j == 0):
                    self.vis_transform(variables)


                if self.log:
                    if (i % self.log_iter == 0) or (i == grad_steps):
                        self.log_result(variables, i)


                if pbar is not None:
                    pbar.progress(i / total_steps)
                else:
                    if i % self.show_iter == 0:
                        t_avg = (time.time() - t_st) / self.show_iter
                        progress_print(
                                'optimize', i, total_steps, 'c', t_avg)
                        t_st = time.time()


            if not is_last_iter:
                loss = self.cma_update(variables, inverted_loss=True)

            self.update_propagation_variable_statistic(variables)

            if np.min(loss) < self._best_loss:
                self._candidate = \
                    variables.transform.t.data[np.argmin(loss)].cpu().detach()
                self._best_loss = np.min(loss)

        candidate_out = variables.output.target.data[np.argmin(loss)]


        if self.log:
            return variables, (self.outs, self.transform_outs, candidate_out),\
                    self.losses

        transform_target = \
            to_grid(torch.stack(variables.output.target.data).cpu())

        transform_out = to_grid(torch.stack(list(self.out.cpu().detach())))

        results = ([transform_out], [transform_target], candidate_out)

        return variables, results, self.loss
