import init_paths

import streamlit as st
import os
import os.path as osp
from PIL import Image
import numpy as np

import torch

from im_utils import center_crop, smart_resize, poisson_blend
from demo import TransformableBasinCMAProjection

from im_utils import to_image, make_grid, to_tensor

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@st.cache()
def load_image(im_path):
    return np.array(Image.open(im_path))


def file_selector(folder_path='./'):
    folder_path = os.path.abspath(folder_path)
    folder_path = st.sidebar.text_input('Search directory:', folder_path)

    valid_file_extensions = ['.jpg', '.png', '.jpeg']
    filenames = [x for x in os.listdir(folder_path)
                 if osp.splitext(x)[1].lower() in valid_file_extensions]

    if len(filenames) == 0:
        st.sidebar.write('No files found')
        return None
    selected_filename = st.sidebar.selectbox('Select a file', filenames)
    return load_image(os.path.join(folder_path, selected_filename))


def upload_file():
    img_file_buffer = st.sidebar.file_uploader("Upload an image",
                                               type=["png", "jpg", "jpeg"])

    if img_file_buffer is None:
        return None

    image = Image.open(img_file_buffer)
    return np.array(image)


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_demo():
    solver = TransformableBasinCMAProjection()
    return solver


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def run_optimization(im, c, num_seeds, max_batch_size, cma_steps, adam_steps,
                     ft_adam_steps, transform_cma_steps, transform_adam_steps):
    variables, outs, losses, transform_fn = \
        solver(im, c,
               num_seeds=num_seeds,
               cma_steps=cma_steps,
               adam_steps=adam_steps,
               finetune_steps=ft_adam_steps,
               transform_cma_steps=transform_cma_steps,
               transform_adam_steps=transform_adam_steps,
               encoder_init=True,
               max_batch_size=max_batch_size,
               remove_cache=False,
               st_progress_bar=True)
    return variables, outs, losses, transform_fn


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def detect(im, detection_method):
    mask, pred_cls_noun, det_cls_noun, misc = \
        solver.auto_detect(im, detection_method)
    mask_overlay_im = (im / 255.) * mask.clip(0.3, 1.0)
    return mask_overlay_im, mask, pred_cls_noun, det_cls_noun, misc


def main():
    st.sidebar.markdown('<h1> pix2latent </h1>',
                        unsafe_allow_html=True)
    st.sidebar.markdown('<p> Interactive demo for: <br><b>Transforming and ' +
                        'Projecting Images to Class-conditional Generative' +
                        ' Networks</b></p>',
                        unsafe_allow_html=True)

    load_option = st.sidebar.selectbox('How do you want to load your image?',
                                       ('upload', 'search'))

    if load_option == 'upload':
        im = upload_file()
    else:
        im = file_selector('./examples')

    if im is None:
        return

    if im.shape != (256, 256, 3):
        st.sidebar.text(
            '(Warning): Image is in the incorrect dimension, ' +
            'automatically cropping and resizing image')
        im = center_crop(im)
        im = smart_resize(im)

    # Go through all detection
    st.sidebar.markdown('<center><h3>Selected image</h3></center>',
                        unsafe_allow_html=True)
    im_view = st.sidebar.radio(
        'masking method', ('bbox', 'segmentation'))
    mask_overlay_im, mask, p_noun, d_noun, misc = detect(im, im_view)
    st.sidebar.image(mask_overlay_im)
    st.sidebar.markdown('<b>Detected class</b>: {}'.format(d_noun),
                        unsafe_allow_html=True)
    st.sidebar.markdown('<b>Predicted class</b>: {}'.format(p_noun),
                        unsafe_allow_html=True)

    misc_nouns = np.array([x[1] for x in misc])
    selected = st.sidebar.selectbox('Change class', misc_nouns)
    selected_cls = misc[np.argwhere(misc_nouns == selected).squeeze()][0]

    # Optimization config
    st.sidebar.markdown('<center> <h3> Optimization config </h3> </center>',
                        unsafe_allow_html=True)

    num_seeds = st.sidebar.slider(
        'Number of seeds',
        min_value=1,
        max_value=18,
        value=18,
        step=1,
    )

    if num_seeds != 18:
        st.sidebar.text('(Warning): PyCMA num_seeds is fixed to 18. ' +
                        'Using Nevergrad implementation instead. May ' +
                        'not work as well.')

    max_batch_size = st.sidebar.slider(
        'Max batch size',
        min_value=1,
        max_value=18,
        value=9,
        step=1,
    )

    cma_steps = st.sidebar.slider(
        'CMA update',
        min_value=1,
        max_value=50,
        value=30,
        step=5,
    )

    adam_steps = st.sidebar.slider(
        'ADAM update',
        min_value=1,
        max_value=50,
        value=30,
        step=5,
    )

    ft_adam_steps = st.sidebar.slider(
        'Final ADAM update',
        min_value=1,
        max_value=1000,
        value=300,
        step=50,
    )

    transform_cma_steps = st.sidebar.slider(
        'Transform CMA update',
        min_value=1,
        max_value=50,
        value=30,
        step=5,
    )

    transform_adam_steps = st.sidebar.slider(
        'Transform ADAM update',
        min_value=1,
        max_value=50,
        value=30,
        step=5,
    )

    start_optimization = st.sidebar.button('Optimize')

    if not start_optimization:
        return

    variables, outs, losses, transform_fn = \
        run_optimization(im, selected_cls, num_seeds, max_batch_size,
                         cma_steps, adam_steps, ft_adam_steps,
                         transform_cma_steps, transform_adam_steps)

    # Collage
    collage_results = to_image(make_grid(outs), cv2_format=False)

    # Blended collage
    t = torch.stack(variables.t.data)[:1]
    inv_ims = []
    for out in outs:
        inv_ims.append(
            transform_fn(out.unsqueeze(0).cpu(), t.cpu(), invert=True))

    inv_collage_results = to_image(make_grid(torch.cat(inv_ims)), cv2_format=False)

    # Show Results
    blended = []

    for x in inv_ims:
        inv_im = to_image(x, cv2_format=False)[0]
        b = poisson_blend(im, mask, inv_im)
        blended.append(to_tensor(b))

    blended = torch.cat(blended)
    blended_collage_results = to_image(make_grid(blended), cv2_format=False)

    st.markdown('<h3> Projection </h3>', unsafe_allow_html=True)
    st.image(collage_results, use_column_width=True)
    st.markdown('<h3> Inverted </h3>', unsafe_allow_html=True)
    st.image(inv_collage_results, use_column_width=True)
    st.markdown('<h3> Poisson blended </h3>', unsafe_allow_html=True)
    st.image(blended_collage_results, use_column_width=True)
    return


solver = load_demo()
main()
