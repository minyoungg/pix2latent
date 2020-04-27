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
                     grad_steps=30, z_sigma=0.5, t_sigma=0.05, log=False):
    """
    Searches for transformation parameter to apply to the target image such
    that it is better invertible. The transformation is optimized in a
    BasinCMA like fashion. Outer loop optimizes transformation using CMA and
    the inner loop is optimized using gradient descent.

    Args:
        model:
            generative model
        transform_fn:
            the transformation search function
        var_manager:
            variable manager
        loss_fn:
            loss function to optimize with
        args:
            arguments used in config
    Returns
        Transformation parameters
    """

    var_manager.optimize_t  = True
    t_outs, outs, step_iter = [], [], 0
    total_steps = meta_steps * grad_steps
    t_cma_opt = CMA(mu=var_manager.t.cpu().numpy()[0], sigma=t_sigma)

    variables = var_manager.init(t_cma_opt.batch_size())

    # We will use the binarized weight for the CMA loss
    mask = binarize(variables.weight.data[0]).unsqueeze(0)
    target = variables.target.data[0].unsqueeze(0)

    for i in range(meta_steps):
        t = t_cma_opt.ask()
        t_params = torch.Tensor(t)

        z = propagate_z(variables, z_sigma)

        variables = var_manager.init(num_seeds=t_cma_opt.batch_size(),
                                     z=z.detach().float().cuda(),
                                     t=t_params.detach().float().cuda())

        for x in variables.t.data:
            x.requires_grad = False

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

            if (step_iter + 1) % 50 == 0:
                progress_print('transform', step_iter + 1, total_steps, 'c')

        t_cma_opt.tell(t, np.min(losses, 0))

        if log and 'target' in other.keys():
            t_out = binarize(other['weight'], 0.3) * other['target']
            t_outs.append(to_image(make_grid(t_out), cv2_format=False))

    t_mu = t_cma_opt.mean()
    return torch.Tensor([t_mu]), t_outs, outs
