"""
Hook functions are trigered in optimization_core.py. These functions should be
in-place operation to keep data variables attached to the optimizers.
"""

import torch



class Clamp():
    """
    clamps the variable by the specified truncation value.

    Args:
        trunc (float): truncation value
        *vars (list): list consisting of tensors to be truncated.
    """

    def __init__(self, trunc):
        self.trunc = trunc
        return


    def __call__(self, vars):
        for v in vars:
            v.data.clamp_(-self.trunc, self.trunc)
        return



class Normalize():
    """
    normalizes the variable to be mean 0 and standard deviation of 1
    this is latent normalization used in StyleGAN2

    Args:
        mu (float): the mean to normalize to [Default: 0.]
        std (float): the standard deviation to normalize to [Default: 1.]
    """

    def __init__(self, mu=0., std=1.):
        self.mu = mu
        self.std = std
        return

    def __call__(self, vars):
        for v in vars:
            mean = v.mean()
            std = v.std()
            v.data.add_(-mean).div_(std)
        return



class NormalPerturb():
    """
    perturbs the data using a normal distribution

    Args:
        sigma (float): the standard deviation of the noise [Default: 0.1]
    """
    def __init__(self, sigma=0.1):
        self.sigma = sigma
        return

    def __call__(self, vars):
        for v in vars:
            v.data.add_(self.sigma * torch.randn_like(v))
        return



class ScheduledNormalPerturb():
    """
    perturbs the data by a normal distribution. simplified version of the
    perturbation used in stylegan2. Decays the perturbation noise starting
    from sigma to 0.

    (stylegan2) > noise * sigma * max(0, 1 - t / args.noise_ramp) ** 2
    (pix2latent) > sigma * max(0, 1 - t) ** 2

    Args:
        sigma (float): the standard deviation of the noise [Default: 0.1]
        max_step (int): at max_step, there is noise perturbation
        pow (int): the power in which to decay the noise.
            (e.g. pow=1 is linear decay, pow=2, squared decay)
    """
    def __init__(self, sigma=0.1, max_step=500, pow=2):
        self.sigma = sigma
        self.max_step = max_step
        self.t = 0
        self.pow = 2
        return

    def __call__(self, vars):
        for v in vars:
            p = self.t / (float(self.max_step) - 1)
            noise_strength = math.pow(self.sigma * max(0, 1 - p), self.pow)
            noise = noise_strength * torch.randn_like(v)
            v.data.add_(noise)
        self.t += 1
        return



class Compose():
    """
    Composes mutliple function hooks by applying the functions sequentially

    Args:
        hook_fns (list): A list containing hook function instances.

    Example:
        >>> Compose(
                Normalize(),
                NormalPerturb(),
                )
    """
    def __init__(self, *hook_fns):
        self.hook_fns = hook_fns
        return

    def __call__(self, vars):
        for fn in self.hook_fns:
            fn(vars)
        return
