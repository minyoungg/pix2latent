# GAN-Projection
**Aligning and Projecting Images to Class-conditional Generative Networks**  
[Minyoung Huh](http://minyounghuh.com/) &nbsp; [Richard Zhang](https://richzhang.github.io/) &nbsp; [Jun-Yan Zhu](https://people.csail.mit.edu/junyanz/) &nbsp; [Sylvain Paris](http://people.csail.mit.edu/sparis/) &nbsp; [Aaron Hertzmann](https://www.dgp.toronto.edu/~hertzman/)  
[[Paper]]() &nbsp; [[Project Page]]()  
<b>arXiv preprint 2020</b> 


## Prerequisites
The code was developed on
- Ubuntu 18.04
- Python 3.7
- Pytorch 1.2.0

We use the BigGAN port from HuggingFaces [github](https://github.com/huggingface/pytorch-pretrained-BigGAN)

## Setup
<b> (1) First install the pytorch dependencies </b>
```bash
pip install -r requirements.txt
```

<b> (2) Clone LPIPS </b>  

The code uses perceptual similarity loss. So we need to clone the LPIPS repo into this directory.
```bash
git clone https://github.com/richzhang/PerceptualSimilarity
```
The path should be in the following format ... or you can set the path manually in `init_paths.py`
```bash
./GAN-Projection/PerceptualSimilarity
```

<b> (3) Encoder weights </b>   
The demo code uses encoder weights to speed up optimization.  
Download the weights from [google drive](https://drive.google.com/drive/folders/1CyDQGBlduBP7lk3WiwazsEViT7VJRnE_?usp=sharing) and place it in the `nets/weights` sub-directory.  

The encoder weight should be in the following path 
```bash
./GAN-Projection/nets/weights/encoder.ckpt
```

## Demo
We provide a demo of our project in [example.ipynb](example.ipynb)

Here is a quick TLDR of the demo
```python
solver = TransformableBasinCMAProjection()
results = solver(im, mask=mask, cls_lbl=class_label)
```

## Developement
To get a glimpse on how to extend the work or invert images on your generative model. Take a look at [demo.py](demo.py)

<b> (a) Optimization </b>
We currently support `GradientOptimizer`, `CMAOptimizer`, `BasinCMAOptimizer`. We also have an experimental optimization methods using [NeverGrad](https://github.com/facebookresearch/nevergrad): `NevergradOptimizer` and `NevergradHybridOptimizer`. You can find the optimizers at [optimizer.py](./optimizer.py).

<b> (b) Transformation </b>
We support simple spatial affine transformation and various color transformations. A full list of transformation functions are available at [transform_functions.py](./utils/transform_functions.py). The transformation optimization details are in [transform.py](transform.py).

