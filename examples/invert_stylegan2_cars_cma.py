import os, os.path as osp
import numpy as np
import argparse

import torch
import torch.nn as nn

from pix2latent.model.stylegan2 import StyleGAN2

from pix2latent import VariableManager, save_variables
from pix2latent.optimizer import CMAOptimizer
from pix2latent.utils import image, video

import pix2latent.loss_functions as LF
import pix2latent.utils.function_hooks as hook
import pix2latent.distribution as dist


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--latent_noise', type=float, default=0.05)
parser.add_argument('--truncate', type=float, default=2.0)
parser.add_argument('--max_minibatch', type=int, default=9)
parser.add_argument('--make_video', action='store_true')
args = parser.parse_args()



### ---- initialize --- ###

model = StyleGAN2(model='cars', search='z')

filename = './images/car-example.png'

target = image.read(filename, as_transformed_tensor=True, im_size=512,
                    transform_style='stylegan')

# we apply a mask since the generated resolution is 384 x 512
loss_mask = torch.zeros((3, 512, 512))
loss_mask[:, 64:-64, :].data += 1.0

weight = loss_mask


fn = filename.split('/')[-1].split('.')[0]
save_dir = f'./results/stylegan2_cars/cma_{fn}'



model = StyleGAN2(search='z')
model = nn.DataParallel(model)
loss_fn = LF.ProjectionLoss()


var_manager = VariableManager()

var_manager.register(
                variable_name='z',
                shape=(512,),
                default=None,
                grad_free=True,
                distribution=dist.TruncatedNormalModulo(
                                            sigma=1.0,
                                            trunc=args.truncate
                                            ),
                var_type='input',
                learning_rate=args.lr,
                hook_fn=hook.Compose(
                            hook.NormalPerturb(sigma=args.latent_noise),
                            hook.Clamp(trunc=args.truncate),
                            )
                )

var_manager.register(
                variable_name='target',
                shape=(3, 512, 512),
                requires_grad=False,
                default=target,
                var_type='output'
                )

var_manager.register(
                variable_name='weight',
                shape=(3, 512, 512),
                requires_grad=False,
                default=weight,
                var_type='output'
                )

var_manager.register(
                variable_name='loss_mask',
                shape=(3, 512, 512),
                requires_grad=False,
                default=loss_mask,
                var_type='output'
                )



### ---- optimize --- ###

opt = CMAOptimizer(
            model, var_manager, loss_fn,
            max_batch_size=args.max_minibatch,
            log=args.make_video
            )

opt.log_resize_factor = 0.5

vars, out, loss = opt.optimize(meta_steps=200, grad_steps=300)



### ---- save results ---- #

vars.loss = loss
os.makedirs(save_dir, exist_ok=True)

save_variables(osp.join(save_dir, 'vars.npy'), vars)

if args.make_video:
    video.make_video(osp.join(save_dir, 'out.mp4'), out)

image.save(osp.join(save_dir, 'target.jpg'), target)
image.save(osp.join(save_dir, 'mask.jpg'), image.binarize(weight))
image.save(osp.join(save_dir, 'out.jpg'), out[-1])
np.save(osp.join(save_dir, 'tracked.npy'), opt.tracked)
