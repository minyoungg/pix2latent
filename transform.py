import numpy as np
import copy
import cv2

import torch
import torch.nn.functional as F

from im_utils import binarize, to_image, make_grid
from variable_manager import sample_truncated_normal, override_variables
from cma_optimizer import CMA
import loss_functions as LF
from misc import to_numpy, progress_print
from optimization_core import step


def propagate_z(variables, z_sigma):
    """
    The variable information is re-used to make new samples. This is used to
    speed up search time. Using a moving average performs similarly.
    """
    z_mu = torch.stack(variables.z.data)
    z_noise = torch.randn_like(z_mu).float().cuda()
    z_mu = z_mu.clamp_(-1.0, 1.0).mean(0, keepdim=True)
    z = z_mu + z_sigma * z_noise
    return z


def search_transform(model, transform_fn, var_manager, loss_fn, meta_steps=30,
                     grad_steps=30, z_sigma=0.5, t_sigma=0.05, log=False,
                     pbar=None):
    """
    Searches for transformation parameter to apply to the target image such
    that it is better invertible. The transformation is optimized in a
    BasinCMA like fashion. Outer loop optimizes transformation using CMA and
    the inner loop is optimized using gradient descent.

    TODO: Clean up the code and integrate with optimzers.py

    Args:
        model:
            generative model
        transform_fn:
            the transformation search function
        var_manager:
            variable manager
        loss_fn:
            loss function to optimize with
        meta_steps:
            Number of CMA updates for transformation parameter
        grad_steps:
            Number of ADAM updates for z, c
        z_sigma:
            Sigma for latent variable resampling
        t_sigma:
            Sigma for transformation parameter
        log:
            Returns intermediate transformation result if True
        pbar:
            Progress bar such TQDM or st.progress()
    Returns
        Transformation parameters
    """

    # -- setup CMA -- #
    var_manager.optimize_t  = True
    t_outs, outs, step_iter = [], [], 0
    total_steps = meta_steps * grad_steps
    t_cma_opt = CMA(mu=var_manager.t.cpu().numpy()[0], sigma=t_sigma)

    if t_cma_opt.batch_size() > var_manager.num_seeds:
        import nevergrad as ng
        print('Number of seeds is less than that required by PyCMA ' +\
              'transformation search. Using Nevergrad CMA instead.')
        batch_size = var_manager.num_seeds
        opt_fn = ng.optimizers.registry['CMA']
        p = ng.p.Array(init=var_manager.t.cpu().numpy()[0])
        p = p.set_mutation(sigma=t_sigma)
        t_cma_opt = opt_fn(parametrization=p, budget=meta_steps)
        variables = var_manager.init(var_manager.num_seeds)
        using_nevergrad = True
    else:
        batch_size = t_cma_opt.batch_size()
        variables = var_manager.init(batch_size)
        using_nevergrad = False

    # Note: we will use the binarized weight for the CMA loss.
    mask = binarize(variables.weight.data[0]).unsqueeze(0)
    target = variables.target.data[0].unsqueeze(0)

    for i in range(meta_steps):

        # -- initialize variable -- #
        if using_nevergrad:
            _t = [t_cma_opt.ask() for _ in range(var_manager.num_seeds)]
            t = np.concatenate([np.array(x.args) for x in _t])
        else:
            t = t_cma_opt.ask()

        t_params = torch.Tensor(t)
        z = propagate_z(variables, z_sigma)
        variables = var_manager.init(num_seeds=batch_size,
                                     z=z.detach().float().cuda(),
                                     t=t_params.detach().float().cuda())

        for x in variables.t.data:
            x.requires_grad = False

        # -- inner update -- #
        losses = []
        for j in range(grad_steps):

            out, _, other = step(model, variables,
                                 loss_fn=loss_fn,
                                 transform_fn=transform_fn,
                                 optimize=True)

            step_iter += 1

            # Compute loss after inverting
            t_params = torch.stack(variables.t.data)
            inv_out = transform_fn(out, t_params, invert=True)

            loss = loss_fn(inv_out,
                           target.repeat(inv_out.size(0), 1, 1, 1),
                           mask.repeat(inv_out.size(0), 1, 1, 1))

            losses.append(to_numpy(loss))
            outs.append(to_image(make_grid(out), cv2_format=False))

            if pbar is not None:
                pbar.progress(step_iter / total_steps)
            else:
                if (step_iter + 1) % 50 == 0:
                    progress_print('transform', step_iter + 1, total_steps, 'c')

        # -- update CMA -- #
        if using_nevergrad:
            for z, l in zip(_t, np.min(losses, 0)):
                t_cma_opt.tell(z, l)
        else:
            t_cma_opt.tell(t, np.min(losses, 0))

        # -- log for visualization -- #
        if log and 'target' in other.keys():
            t_out = binarize(other['weight'], 0.3) * other['target']
            t_outs.append(to_image(make_grid(t_out), cv2_format=False))

    if using_nevergrad:
        t_mu = np.array(t_cma_opt.provide_recommendation().value)
    else:
        t_mu = t_cma_opt.mean()
    return torch.Tensor([t_mu]), t_outs, outs
