import os, os.path as osp
import numpy as np
import argparse

import torch
import torch.nn as nn

from pix2latent.model.biggan import BigGAN

from pix2latent import VariableManager, save_variables
from pix2latent.optimizer import NevergradOptimizer
from pix2latent.utils import image, video

import pix2latent.loss_functions as LF
import pix2latent.utils.function_hooks as hook
import pix2latent.distribution as dist


parser = argparse.ArgumentParser()
parser.add_argument('--ng_method', type=str, default='CMA')
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--latent_noise', type=float, default=0.05)
parser.add_argument('--truncate', type=float, default=2.0)
parser.add_argument('--num_samples', type=int, default=4)
parser.add_argument('--max_minibatch', type=int, default=9)
parser.add_argument('--make_video', action='store_true')
args = parser.parse_args()



### ---- initialize necessary --- ###

# (1) pretrained generative model
model = BigGAN().cuda().eval()

# (2) variable creator
var_manager = VariableManager()

# (3) default l1 + lpips loss function
loss_fn = LF.ProjectionLoss()

filename = './images/dog-example-153.jpg'
mask_filename = './images/dog-example-153-mask.jpg'
class_lbl = 182

target = image.read(filename, as_transformed_tensor=True, im_size=256)
weight = image.read(mask_filename, as_transformed_tensor=True, im_size=256)
weight = ((weight + 1.) / 2.).clamp_(0.3, 1.0)

fn = filename.split('/')[-1].split('.')[0]
save_dir = f'./results/biggan_256/ng_{fn}'


var_manager = VariableManager()
loss_fn = LF.ProjectionLoss()


# (4) define input output variable structure. the variable name must match
# the argument name of the model and loss function call

var_manager.register(
            variable_name='z',
            shape=(128,),
            grad_free=True,
            distribution=dist.TruncatedNormalModulo(
                                sigma=1.0,
                                trunc=args.truncate
                                ),
            var_type='input',
            learning_rate=args.lr,
            hook_fn=hook.Clamp(args.truncate)
            )

var_manager.register(
            variable_name='c',
            shape=(128,),
            requires_grad=True,
            default=model.get_class_embedding(class_lbl)[0],
            var_type='input',
            learning_rate=0.01,
            )

var_manager.register(
            variable_name='target',
            shape=(3, 256, 256),
            requires_grad=False,
            default=target,
            var_type='output'
            )

var_manager.register(
            variable_name='weight',
            shape=(3, 256, 256),
            requires_grad=False,
            default=weight,
            var_type='output'
            )


### ---- optimize --- ###

opt = NevergradOptimizer(
         args.ng_method, model, var_manager, loss_fn,
         max_batch_size=args.max_minibatch,
         log=args.make_video
         )

vars, out, loss = \
    opt.optimize(num_samples=args.num_samples, meta_steps=1000, grad_steps=300)


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
