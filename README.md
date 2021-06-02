# pix2latent: framework for inverting images into generative models
Framework for inverting images. Codebase used in:

**Transforming and Projecting Images into Class-conditional Generative Networks**  
**[project page](https://minyoungg.github.io/pix2latent/) |   [paper](http://people.csail.mit.edu/minhuh/papers/pix2latent/arxiv_v2.pdf)**     
[Minyoung Huh](http://minyounghuh.com/) &nbsp; [Richard Zhang](https://richzhang.github.io/) &nbsp; [Jun-Yan Zhu](https://people.csail.mit.edu/junyanz/) &nbsp; [Sylvain Paris](http://people.csail.mit.edu/sparis/) &nbsp; [Aaron Hertzmann](https://www.dgp.toronto.edu/~hertzman/)  
MIT CSAIL &nbsp; Adobe Research  
**ECCV 2020 (oral)**

![](./assets/overview.gif)

```
@inproceedings{huh2020ganprojection,
    title = {Transforming and Projecting Images to Class-conditional Generative Networks},
    author = {Minyoung Huh and Richard Zhang, Jun-Yan Zhu and Sylvain Paris and Aaron Hertzmann},
    booktitle = {ECCV},
    year = {2020}
}
```

**NOTE [8/25/20]** The codebase has been renamed from `GAN-Transform-and-Project` to `pix2latent`, and also refactored to make it easier to use and extend to any generative model beyond `BigGAN`. To access the original codebase refer to the `legacy` branch. 

# Example results
<b>All results below are without fine-tuning.</b>

<p align="center"><b> BigGAN (z-space) - ImageNet (256x256) </b></p>

![](./assets/biggan_comparison.png)

<p align="center"><b> StyleGAN2 (z-space) - LSUN Cars (384x512) </b></p>

![](./assets/stylegan2_cars.png)

<p align="center"><b> StyleGAN2 (z-space) - FFHQ (1024x1024) </b></p>

![](./assets/stylegan2_ffhq.png)


## Prerequisites
The code was developed on
- Ubuntu 18.04
- Python 3.7
- PyTorch 1.4.0

## Getting Started
- <b>Install PyTorch</b>  
  Install the correct [PyTorch version](https://pytorch.org/) for your machine
  
- <b>Install the python dependencies</b>  
  Install the remaining dependencies via
  ```bash
  pip install -r requirements.txt
  ```
- <b>Install pix2latent</b>
  ```bash
  git clone https://github.com/minyoungg/pix2latent
  cd pix2latent
  pip install .
  ```

## Examples
We provide several demo codes in `./examples/` for both [`BigGAN`](https://arxiv.org/abs/1809.11096) and [`StyleGAN2`](https://arxiv.org/abs/1912.04958). Note that the codebase has been tuned and developed on `BigGAN`. 

```bash
> cd examples
> python invert_biggan_adam.py --num_samples 4
```

Using the `make_video` flag will save the optimization trajectory as a video.
```bash
> python invert_biggan_adam.py --make_video --num_samples 4
```

**(slow)** To optimize with `CMA-ES` or `BasinCMA`, we use [PyCMA](https://github.com/CMA-ES/pycma). Note that the PyCMA version of CMA-ES has a predefined number of samples to jointly evaluate (18 for BigGAN) and (22 for StyleGAN2). 
```bash
> python invert_biggan_cma.py 
> python invert_biggan_basincma.py 
```

**(fast)** Alternatively CMA-ES in [Nevergrad](https://github.com/facebookresearch/nevergrad) provides sample parallelization so you can set your own number of samples. Although this runs faster, we have observed the performance to be slightly worse. **(warning: performance depends on num_samples)**.
```bash
> python invert_biggan_nevergrad.py --ng_method CMA --num_samples 4
> python invert_biggan_hybrid_nevergrad.py --ng_method CMA --num_samples 4
```

Same applies to `StyleGAN2`. See `./examples/` for extensive list of examples.


### Template pseudocode
```python
import torch, torch.nn as nn
import pix2latent.VariableManger
from pix2latent.optimizer import GradientOptimizer

# load your favorite model
class Generator(nn.Module):
    ...
    
    def forward(self, z):
        ...
        return im

model = Generator() 

# define your loss objective .. or use the predefined loss functions in pix2latent.loss_functions
loss_fn = lambda out, target: (target - out).abs().mean()

# tell the optimizer what the input-output relationship is
vm = VariableManager()
vm.register(variable_name='z', shape=(128,), var_type='input')
vm.register(variable_name='target', shape(3, 256, 256), var_type='output')

# setup optimizer
opt = GradientOptimizer(model, vm, loss_fn)

# optimize
vars, out, loss = opt.optimize(num_samples=1, grad_steps=500)

```


### detailed usage

#### `pix2latent`

| Command | Description |
| --- | --- |
| `pix2latent.loss_function` | predefined loss functions |
| `pix2latent.distribution` | distribution functions used to initialize variables |


#### `pix2latent.VariableManger`
class variable for managing variables. variable manager instance is initialized by
```
var_man = VariableManager()
```

| Method | Description |
| --- | --- |
| `var_man.register(...)` | registers variable. this variable is created when `initialize` is called |
| `var_man.unregister(...)` | removes a variable that is already registered |
| `var_man.edit_variable(...)` | edits existing variable |
| `var_man.initialize(...)` | initializes variable from defined specification |


#### `pix2latent.optimizer`
| Command | Description |
| --- | --- |
| `pix2latent.optimizer.GradientOptimizer` | gradient-based optimizer. defaults to optimizer defined in `pix2latent.VariableManager`|
| `pix2latent.optimizer.CMAOptimizer` | uses CMA optimization to search over latent variables `z`|
| `pix2latent.optimizer.BasinCMAOptimizer` | uses BasinCMA optimization. a combination of CMA and gradient-based optimization|
| `pix2latent.optimizer.NevergradOptimizer` | uses [`Nevergrad`](https://github.com/facebookresearch/nevergrad) library for optimization. supports most gradient-free optimization method implemented in [`Nevergrad`](https://github.com/facebookresearch/nevergrad) |
| `pix2latent.optimizer.HybridNevergradOptimizer` | uses hybrid optimization by alternating gradient and gradient-free optimization provided by [`Nevergrad`](https://github.com/facebookresearch/nevergrad)|


#### `pix2latent.transform`
| Command | Description |
| --- | --- |
| `pix2latent.SpatialTransform` | spatial transformation function, used to optimize for image scale and position |
| `pix2latent.TransformBasinCMAOptimizer` | BasinCMA-like optimization method used to search for image transformation |


#### `pix2latent.util`
| Command | Description |
| --- | --- |
| `pix2latent.util.image` | utility for image pre and post processing |
| `pix2latent.util.video` | utility for video (e.g. saving videos) |
| `pix2latent.util.misc` | miscellaneous functions  |
| `pix2latent.util.function_hooks` | function hooks that can be attached to variables in the optimization loop. (e.g. `Clamp`, `Perturb`) |

#### `pix2latent.model`
| Command | Description |
| --- | --- |
| `pix2latent.model.BigGAN` | BigGAN model wrapper. Uses implementation by [`huggingface`](https://github.com/huggingface/pytorch-pretrained-BigGAN) using the official weights|
| `pix2latent.model.StyleGAN2` | StyleGAN2 model wrapper. Uses PyTorch implementation by [`rosinality`](https://github.com/rosinality/stylegan2-pytorch) using the official weights|

#### `pix2latent.edit`
| Command | Description |
| --- | --- |
| `pix2latent.edit.BigGANLatentEditor` | BigGAN editor. Simple interface to edit class and latent variables using oversimplified version of [`GANSpace`](https://github.com/harskish/ganspace) |


