import torch

from pix2latent.variable_manager import split_vars


def step(model, vars, loss_fn, optimize=True, max_batch_size=9):
    """
    The step function for model evaluation.

    Args:
        vars: variable object generated from variable_manager.
        loss_fn: loss function to compute the loss. The loss function takes 3
            arguments output, target, mask.
        optimize: if False, does not compute gradient
        max_batch_size: if the number of seeds in vars is great max_batch_size
            it will get chunked into mini-batches.

    Returns:
        outs: the output of the model
        indiv_losses: the loss for each sample.
        misc_return: a dictionary containing miscellaneous stuff to log

    """

    outs, indiv_losses, targets, weights = [], [], [], []

    for s_iter, _vars in enumerate(split_vars(vars, size=max_batch_size)):

        def closure():
            global out, loss, rec_loss, per_loss, _target, _weight
            b_sz = _vars.num_samples

            target_args = \
                {k: torch.stack(v.data) for k, v in _vars.output.items()}


            if optimize:
                _vars.opt.zero_grad()


            # (1) apply function hooks (e.g. variable clamping)
            for var_name, var_dict in _vars.input.items():
                if var_dict.hook_fn is not None:
                    var_dict.hook_fn(var_dict.data)


            # (2) forward pass
            input_args = \
                    {k: torch.stack(v.data) for k, v in _vars.input.items()}

            out = model(**input_args)


            # (3) compute loss
            loss = loss_fn(out, **target_args).view(b_sz, -1).mean(1)

            if optimize:
                loss.mean().backward()

            loss = loss.detach().cpu().numpy()


        # (4) optimize
        if optimize:
            _vars.opt.step(closure)
        else:
            with torch.no_grad():
                _vars.opt.step(closure)

        if optimize:
            _vars.opt.zero_grad()


        outs.extend(out.detach())
        indiv_losses.extend(loss)


    misc_return = {}
    return torch.stack(outs), indiv_losses, misc_return
