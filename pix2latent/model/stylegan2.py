import os, os.path as osp
import sys

import torch
import torch.nn as nn


dir_path = os.path.dirname(os.path.realpath(__file__))
stylegan2_dir = osp.join(dir_path, 'stylegan2-pytorch')


def clone_repo():
    """
    cloning pytorch version from the following repo
    `https://github.com/rosinality/stylegan2-pytorch`
    which uses weights converted from the original paper
    `https://github.com/NVlabs/stylegan2`
    """

    if osp.exists(stylegan2_dir):
        return

    print(f'cloning StyleGAN2 repository to {stylegan2_dir}')
    git_url = "https://github.com/rosinality/stylegan2-pytorch"
    os.system(f'git clone {git_url} {stylegan2_dir}')

    assert osp.exists(stylegan2_dir), 'failed to clone from git.'
    return



def download_checkpoint(url, save_path):
    import gdown

    if osp.exists(save_path):
        return

    print('downloading checkpoint from google drive ..')
    gdown.download(url, save_path, quiet=False)

    assert osp.exists(stylegan2_dir), \
        'failed to download checkpoint from Google drive. download it ' + \
        'manually from google drive.'
    return




#def download()

ckpts_dicts = {
    'cars': {
        'url': 'https://drive.google.com/uc?export=download&id=1t14Ld7zAbpsZ3nd8cnoCisC50lpGFR_V',
        'file_path': f'{stylegan2_dir}/stylegan2-car-config-f.pt',
        'im_dim': 512,
    },
    'ffhq': {
        'url': 'https://drive.google.com/uc?export=download&id=19LtTz6kQPN2i_mCU7wyyaX2BpdO4Pq3w',
        'file_path': f'{stylegan2_dir}/stylegan2-ffhq-config-f.pt',
        'im_dim': 1024,
    }
}



class StyleGAN2(nn.Module):
    def __init__(self, model='cars', search='z'):
        super(StyleGAN2, self).__init__()

        # clone repo if it doesnt exist
        clone_repo()

        sys.path.append(stylegan2_dir)
        from model import Generator

        # download weights
        ckpt_meta = ckpts_dicts[model]
        self.im_res = im_res = ckpt_meta['im_dim']

        download_checkpoint(ckpt_meta['url'], ckpt_meta['file_path'])

        # load checkpoint
        self.model = Generator(im_res, 512, 8, channel_multiplier=2).cuda()
        checkpoint = torch.load(ckpt_meta['file_path'])
        self.model.load_state_dict(checkpoint['g_ema'])


        self.noise_shape = []
        for i in range(self.model.num_layers):
            noise = getattr(self.model.noises, f'noise_{i}')
            self.noise_shape.append(list(noise.size()))

        with torch.no_grad():
            n_mean_latent = 4096

            if search == 'z':
                self.mean_latent = self.model.mean_latent(n_mean_latent)

            elif search == 'w+':
                noise_sample = torch.randn(n_mean_latent, 512).cuda()
                latent_out = self.model.style(noise_sample)
                self.latent_mean = latent_out.mean(0)
                latent_std = (latent_out - self.latent_mean).pow(2).sum()
                self.latent_std = (latent_std / n_mean_latent) ** 0.5

        self.search = search
        return


    def __call__(self, z, noises=None, truncation=1.0):
        if self.search == 'w+':
            return self.forward_w(z, noises)
        return self.forward_z(z)


    def forward_z(self, z, truncation=1.0):
        out = self.model(
                [z],truncation=1.0, truncation_latent=self.mean_latent)[0]
        return out.clamp_(-1.0, 1.0)


    def forward_w(self, z, noises, truncation=1.0):
        noises = self.reshape_noise(noises)
        out = self.model([z], input_is_latent=True, noise=noises)[0]
        return out.clamp_(-1.0, 1.0)


    def reshape_noise(self, z):

        st_idx = 0
        noises = []
        for d in self.noise_shape:
            en_idx = st_idx + (d[-2] * d[-1])
            noises.append(z[:, st_idx:en_idx].reshape(-1, 1, d[-2], d[-1]))
            st_idx = en_idx

        assert z.size(1) == en_idx
        return noises
