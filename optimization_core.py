import torch
import numpy as np
import cv2

from variable_manager import split_vars
from im_utils import to_image


def step(model, vars, loss_fn, transform_fn=None, optimize=True,
         max_batch_size=9):
    """
    The step function for model evaluation.

    Args:
        vars:
            variable object generated from variable_manager.
        loss_fn:
            loss function to compute the loss. The loss function takes 3
            arguments output, target, mask.
        transform_fn:
            transformation function
        optimize:
            if False, does not compute gradient
        max_batch_size:
            if the number of seeds in vars is great max_batch_size it will
            get chunked into mini-batches.

    Returns:
        outs:
            The output of the model
        indiv_losses:
            The loss for each seed.
        misc_return:
            A dictionary containing stuff that one might want to log.
    """

    outs, indiv_losses, targets, weights = [], [], [], []

    for s_iter, _vars in enumerate(split_vars(vars, size=max_batch_size)):

        def closure():
            global out, loss, rec_loss, per_loss, _target, _weight
            b_sz = len(_vars.z.data)

            _target = torch.stack(vars.target.data).repeat(b_sz, 1, 1, 1)
            _weight = torch.stack(vars.weight.data).repeat(b_sz, 1, 1, 1)


            if hasattr(_vars, 't'):
                t = torch.stack(_vars.t.data)

                if len(_vars.t.data) == 1:
                    t = t.repeat([b_sz, 1])

                _target = transform_fn(_target, t)
                _weight = transform_fn(_weight, t, only_spatial=True)


            if optimize:
                _vars.opt.zero_grad()


            # (1) clamp z
            for i in range(len(_vars.z.data)):
                _vars.z.data[i].data.clamp_(-2.0, 2.0)


            # (2) forward pass
            z, cv = torch.stack(_vars.z.data), torch.stack(_vars.cv.data)
            out = model(z=z, c=cv, truncation=1.0)


            # (3) compute loss
            loss = loss_fn(out, _target, _weight).view(b_sz, -1).mean(1)

            if optimize:
                loss.mean().backward()

            loss = loss.detach().cpu().numpy()


        # (4) optimize
        _vars.opt.step(closure)

        outs.extend(out.detach())
        indiv_losses.extend(loss)

        if hasattr(_vars, 't'):
            targets.extend(_target.detach().cpu())
            weights.extend(_weight.detach().cpu())

    misc_return = {}
    if hasattr(_vars, 't'):
        misc_return['target'] = torch.stack(targets)
        misc_return['weight'] = torch.stack(weights)

    return torch.stack(outs), indiv_losses, misc_return
