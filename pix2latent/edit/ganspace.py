import torch
import torch.nn.functional as F


def biggan_components(model, class_lbl, num_components=32, num_samples=12800,
                      feat_size=128, method='sgd'):
    """
    Args:
        model: BigGAN model instance
        num_components: number of PCA components
        num_samples: number of samples to estimate PCA
        feat_size: feature size of BigGAN
        method: method for solving for z-space principal components

    Quick and dirty implementation of:
    Erik Härkönen et al. - GANSpace: Discovering Interpretable GAN Controls
    https://github.com/harskish/ganspace
    https://arxiv.org/abs/2004.02546

    WARNING: Do not use this as a benchmark for the work above.
    """
    assert method in ['sgd', 'lstsq']


    # Compute features and run PCA
    with torch.no_grad():
        z = torch.randn(num_samples, feat_size).cuda()

        if type(class_lbl) is int:
            c = model.get_class_embedding(class_lbl)
        else:
            c = class_lbl

        c = c.repeat(z.size(0), 1)

        feat = model.generator.gen_z(torch.cat([z, c], 1))
        feat_mean = feat.mean(0).unsqueeze(0)

        u, s, v = torch.pca_lowrank(feat, q=num_components)
        x = torch.mm(feat - feat_mean, v)


    if method == 'sgd':
        # convex problem, should converge fast
        u = torch.nn.parameter.Parameter(
        torch.randn(feat_size, num_components, requires_grad=True).cuda())

        lr = 1
        opt = torch.optim.Adam([u], lr=lr)

        for i in range(100):
            opt.zero_grad()
            loss = ((z - torch.mm(x, u.t())) ** 2).mean()
            loss.backward()
            opt.step()

            for param_group in opt.param_groups:
                param_group['lr'] = param_group['lr'] * 0.98

        u = F.normalize(u, p=2, dim=1)

    elif method == 'lstsq':
        # torch.lstsq is throwing error, using SGD for now
        raise NotImplementedError()

    return u.permute(1, 0)
