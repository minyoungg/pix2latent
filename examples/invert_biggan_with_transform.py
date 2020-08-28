
import os, os.path as osp
import numpy as np
import argparse

import torch
import torch.nn as nn

from pix2latent.model import BigGAN

from pix2latent import VariableManager, save_variables
from pix2latent.optimizer import GradientOptimizer, CMAOptimizer, \
                BasinCMAOptimizer, NevergradOptimizer, HybridNevergradOptimizer
from pix2latent.transform import TransformBasinCMAOptimizer, SpatialTransform
from pix2latent.utils import image, video

import pix2latent.loss_functions as LF
import pix2latent.utils.function_hooks as hook
import pix2latent.distribution as dist


parser = argparse.ArgumentParser()
parser.add_argument('--fp', type=str,
                    default='./images/dog-example-153.jpg')
parser.add_argument('--mask_fp', type=str,
                    default='./images/dog-example-153-mask.jpg')
parser.add_argument('--class_lbl', type=int, default=153)
parser.add_argument('--method', type=str, required=True,
                    choices=['gradfree', 'hybrid'])
parser.add_argument('--ng_method', type=str, default='CMA')
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--latent_noise', type=float, default=0.05)
parser.add_argument('--truncate', type=float, default=2.0)
parser.add_argument('--make_video', action='store_true')
parser.add_argument('--max_minibatch', type=int, default=9)
parser.add_argument('--num_samples', type=int, default=9)
args = parser.parse_args()



### ---- initialize necessary --- ###

# (1) pretrained generative model
model = BigGAN().cuda().eval()

# (2) variable creator
var_manager = VariableManager()

# (3) default l1 + lpips loss function
loss_fn = LF.ProjectionLoss()


target = image.read(args.fp, as_transformed_tensor=True, im_size=256)
weight = image.read(args.mask_fp, as_transformed_tensor=True, im_size=256)
weight = ((weight + 1.) / 2.).clamp_(0.3, 1.0)

fn = args.fp.split('/')[-1].split('.')[0]
save_dir = f'./results/biggan_256/{args.method}_{fn}_w_transform'


var_manager = VariableManager()
loss_fn = LF.ProjectionLoss()


# (4) define input output variable structure. the variable name must match
# the argument name of the model and loss function call

var_manager.register(
            variable_name='z',
            shape=(128,),
            distribution=dist.TruncatedNormalModulo(
                                sigma=1.0,
                                trunc=args.truncate
                                ),
            var_type='input',
            learning_rate=args.lr,
            hook_fn=hook.Clamp(args.truncate),
            )

var_manager.register(
            variable_name='c',
            shape=(128,),
            default=model.get_class_embedding(args.class_lbl)[0],
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



### ---- optimize (transformation) ---- ####

target_transform_fn = SpatialTransform(pre_align=mask)
weight_transform_fn = SpatialTransform(pre_align=mask)

tranform_params = target_transform_fn.get_default_param(as_tensor=True)

var_manager.register(
            variable_name='t',
            shape=tuple(tranform_params.size()),
            requires_grad=False,
            var_type='transform',
            grad_free=True,
            )

t_opt = TransformBasinCMAOptimizer(
            model, var_manager, loss_fn, max_batch_size=8, log=args.make_video)

# this tells the optimizer to apply transformation `target_transform_fn`
#  with parameter `t` on the variable `target`
t_opt.register_transform(target_transform_fn, 't', 'target')
t_opt.register_transform(weight_transform_fn, 't', 'weight')

# (highly recommended) speeds up optimization by propating information
t_opt.set_variable_propagation('z')


t_vars, (t_out, t_target, t_candidate), t_loss = \
                t_opt.optimize(meta_steps=50, grad_steps=10)


os.makedirs(save_dir, exist_ok=True)

if args.make_video:
    video.make_video(osp.join(save_dir, 'transform_out.mp4'), t_out)
    video.make_video(osp.join(save_dir, 'transform_target.mp4'), t_target)

image.save(osp.join(save_dir, 'transform_out.jpg'), t_out[-1])
image.save(osp.join(save_dir, 'transform_target.jpg'), t_target[-1])
image.save(osp.join(save_dir, 'transform_candidate.jpg'), t_candidate)

np.save(osp.join(save_dir, 'transform_tracked.npy'),
        {'t': t_opt.transform_tracked})

t = t_opt.get_candidate()

var_manager.edit_variable('t', {'default': t, 'grad_free': False})
var_manager.edit_variable('z', {'learning_rate': args.lr})


del t_opt, t_vars, t_out, t_target, t_candidate, t_loss
model.zero_grad()
torch.cuda.empty_cache()



### ---- optimize (latent) ---- ###

if args.method == 'adam':
    var_manager.edit_variable('z', {'grad_free': False})
    opt = GradientOptimizer(
            model, var_manager, loss_fn,
            max_batch_size=args.max_minibatch,
            log=args.make_video
            )
    opt.register_transform(target_transform_fn, 't', 'target')
    opt.register_transform(weight_transform_fn, 't', 'weight')
    vars, out, loss = opt.optimize(num_samples=args.num_samples, grad_steps=500)


elif args.method == 'cma':
    var_manager.edit_variable('z', {'grad_free': True})
    opt = CMAOptimizer(
                model, var_manager, loss_fn,
                max_batch_size=args.max_minibatch,
                log=args.make_video
                )
    opt.register_transform(target_transform_fn, 't', 'target')
    opt.register_transform(weight_transform_fn, 't', 'weight')
    vars, out, loss = opt.optimize(meta_steps=200, grad_steps=300)


elif args.method == 'basincma':
    var_manager.edit_variable('z', {'grad_free': True})
    opt = BasinCMAOptimizer(
                model, var_manager, loss_fn,
                max_batch_size=args.max_minibatch,
                log=args.make_video
                )
    opt.register_transform(target_transform_fn, 't', 'target')
    opt.register_transform(weight_transform_fn, 't', 'weight')
    vars, out, loss = \
            opt.optimize(meta_steps=30, grad_steps=30, last_grad_steps=300)

elif args.method == 'ng':
    var_manager.edit_variable('z', {'grad_free': True})
    opt = NevergradOptimizer(
                 args.ng_method, model, var_manager, loss_fn,
                 max_batch_size=args.max_minibatch,
                 log=args.make_video
                 )

    opt.register_transform(target_transform_fn, 't', 'target')
    opt.register_transform(weight_transform_fn, 't', 'weight')
    vars, out, loss = opt.optimize(
                            num_samples=args.num_samples,
                            meta_steps=1000, grad_steps=300
                            )

elif args.method == 'hybridng':
    opt = HybridNevergradOptimizer(
                    args.ng_method, model, var_manager, loss_fn,
                    max_batch_size=args.max_minibatch,
                    log=args.make_video
                    )

    opt.register_transform(target_transform_fn, 't', 'target')
    opt.register_transform(weight_transform_fn, 't', 'weight')
    vars, out, loss = opt.optimize(
                            num_samples=args.num_samples, meta_steps=30,
                            grad_steps=50, last_grad_steps=300,
                            )


### ---- save results ---- #

vars.loss = loss
save_variables(osp.join(save_dir, 'vars.npy'), vars)

if args.make_video:
    video.make_video(osp.join(save_dir, 'out.mp4'), out)

image.save(osp.join(save_dir, 'target.jpg'), target)
image.save(osp.join(save_dir, 'mask.jpg'), image.binarize(weight))
image.save(osp.join(save_dir, 'out.jpg'), out[-1])
np.save(osp.join(save_dir, 'tracked.npy'), opt.tracked)
