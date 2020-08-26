import torch



class TruncatedNormalModulo():
    """
    Truncated normal distribution. Values that lie outside the truncation value
    are set to the remainder using float-modulo (fmod).

    Args:
        num_samples (int): number of samples to draw from the distribution
        sigma (float): the ``standard deviation`` of the normal distribution.
            computed in the form sigma * N(0, I). [Default: 1.0]
        trunc (float): fmod truncation value. [Default: 2.0]

        -- Distribution Args --

        *num_samples (int): number of samples to draw from the distribution
        *shape (int): the shape of the sample

    """
    def __init__(self, mu=0., sigma=1., trunc=2.):
        if type(mu) in [int, float]:
            self.mu = mu
        else:
            self.mu = mu.detach().cpu()
        self.sigma = 1.0
        self.trunc = 2.0
        return

    def __call__(self, num_samples, shape):
        with torch.no_grad():
            _x = self.sigma * torch.randn((num_samples, *shape))
            return torch.fmod(_x + self.mu, self.trunc)




def truncated_clamp_normal(sigma=1.0, trunc=2.0):
    """
    Truncated normal distribution. Values that lie outside the truncation value
    are hard set to the truncation max value.

    Args:
        sigma (float): the ``standard deviation`` of the normal distribution.
            computed in the form sigma * N(0, I). [Default: 1.0]
        trunc (float): fmod truncation value. [Default: 2.0]

        -- Distribution Args --

        *num_samples (int): number of samples to draw from the distribution
        *shape (int): the shape of the sample

    """
    def _dist_fn(num_samples, shape):
        with torch.no_grad():
            return (sigma * torch.randn((samples, *shape)))._clamp(-trunc, trunc)
    return _dist_fn



def normal(sigma=1.0):
    """
    Normal distribution.

    Args:
        sigma (float): the ``standard deviation`` of the normal distribution.
            computed in the form sigma * N(0, I). [Default: 1.0]

        -- Distribution Args --

        *num_samples (int): number of samples to draw from the distribution
        *shape (int): the shape of the sample
    """
    def _dist_fn(num_samples, shape):
        with torch.no_grad():
            return (sigma * torch.randn((num_samples, *shape)))
    return _dist_fn
