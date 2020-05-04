from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import init_paths

import torch
import torch.nn as nn

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from variable_manager import VariableManager
from misc import to_numpy
import loss_functions as LF
import copy
import optimizers
import transform_functions as TF
from im_utils import to_image, make_grid, binarize
from biggan import BigGAN

from transform_utils import compute_pre_alignment
from transform import search_transform
from dataset_misc import \
    IMAGENET_LABEL_TO_NOUN, IMAGENET_LABEL_TO_WNID, COCO_INSTANCE_CATEGORY_NAMES
import imagenet_tools

from classifier import Classifier
from encoder import load_biggan_encoder
from detector import Detector



class TransformableBasinCMAProjection():
    """
    A demo code for projecting images into BigGAN using transformation ssearch
    and BasinCMA. This demo code optimizes for runtime not quality.

    NOTE
    If runtime is not a concern set BasinCMA iteration to 50 CMA updates and
    50 ADAM updates and lpips_net='vgg'. Results in the paper were generated
    with 30 CMA updates and 30 ADAM updates with alex-lpips. To speed things up
    use 10 CMA updates and 10 ADAM updates with alex-lpips the encoder.
    """

    def __init__(self, image_shape=(256, 256, 3), lpips_net='alex'):
        """
        Initialize network and perceptual loss. Currently only supports
        class-conditional ImageNet models.

        Args
            image_shape:
                The output image dimension of the model.
            lpips_net:
                The perceptual loss model.

        TODO
            Support custom generative model.
        """
        self.model = nn.DataParallel(BigGAN()).cuda().float().eval()
        self.orig_state_dict = self.model.state_dict()
        self.image_shape = image_shape

        self.rloss_fn = LF.ReconstructionLoss()
        self.ploss_fn = LF.PerceptualLoss(net=lpips_net, precision='float')
        self.loss_fn = lambda x, y, m: \
                        self.rloss_fn(x, y, m) + 10. * self.ploss_fn(x, y, m)
        return


    def prepare_input(self, im, cls_lbl, mask=None, mask_threshold=0.3):
        """
        Converts input variables into model ready format.

        Args:
            im:
                An image of shape HWC that we want to project. The image should
                be in numpy format.
            cls_lbl
                The class label integer corresponding to the ImageNet class.
            mask
                A single channel mask HW1 corresponding to the how we want to
                weight the loss. If mask is not provided, weights are computed
                on the full image.
            mask_threshold
                Mask is thresholded such the range is [mask_threshold, 1.0]
        Returns:
            im_tensor:
                A normalized image tensor.
            cv:
                A 1-dimensional tensor containig the continuous embedding .f
                the class label.
            mask_tesnor:
                A mask tensor if provided.
        """

        assert im.shape == self.image_shape, \
            'expected shape {} but got {}'.format(self.image_shape, im.shape)

        assert type(cls_lbl) == int, 'expected cls_lbl to be an integer'

        if np.max(im) > 1.0 + 1e-6:
            im = im / 255.

        if mask is not None:
            assert mask.shape[:2] == im.shape[:2], \
                'im and mask have different spatial dimensions {} vs {}'.format(
                mask.shape, im.shape)

            assert mask.shape[2] == 1, \
                'expected channel size 1 but got {}'.format(mask.shape[2])

            assert np.min(mask) > -1e-6, \
                'mask has a negative value: {}'.format(np.min(mask))

            mask = mask.reshape(1, 1, mask.shape[0], mask.shape[1])

            if np.max(mask) > 1.0 + 1e-6:
                mask = mask / 255.
        else:
            mask = np.ones((1, 1, 256, 256))

        with torch.no_grad():
            c = torch.zeros(1, 1000).float().cuda()
            c[:, cls_lbl] = 1.0
            cv = self.model(c=c, embed_class=True)

        # To tensor
        mask_tensor = torch.from_numpy(mask).clamp_(mask_threshold, 1.0)
        mask_tensor = mask_tensor.float().cuda()

        im_tensor = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0)
        im_tensor = 2.0 * (im_tensor - 0.5)
        im_tensor = im_tensor.float().cuda()
        return im_tensor, cv, mask_tensor


    def __call__(self,
                 im,
                 cls_lbl=None,
                 mask=None,
                 num_seeds=18,
                 cma_steps=10,
                 adam_steps=10,
                 finetune_steps=500,
                 transform_cma_steps=30,
                 transform_adam_steps=30,
                 encoder_init=True,
                 max_batch_size=9,
                 log=False,
                 remove_cache=True,
                 st_progress_bar=False
                 ):
        """
        Projects image using BasinCMA and transformation search.

        Args:
            im
                The image you want to ivert.
            cls_lbl
                Integer indicating the class label of the image. If cls_lbl
                is None, it will use a classifier to predict the class.
            mask:
                The mask is used to weight the loss. If mask is None,
                computes the loss on the full image.
                If mask = 'seg': attempts to use MaskRCNN to get segmentation
                If mask = 'bbox': attempts to use MaskRCNN to get bbox.
                To check if the predicted label and the mask is behaving the
                way you want. Run model.auto_detect().
            num_seeds.
                Number of seeds to optimize.
            cma_steps:
                Number of outer loop CMA optimization for the latent variable.
            adam_steps:
                Number of adam updates per CMA optimization.
            finetune_steps:
                Number of adam updates to apply after the final CMA update
            transform_cma_steps:
                Number of CMA updates for transformation.
            transform_adam_steps:
                Number of ADAM updates per CMA update for transformation.
            encoder_init:
                If True, uses an encoder to warmstart the search space.
            max_batch_size:
                If num_seeds > max_batch_size, it will split the mini-batch
                into num_seeds // max_batch_size mini-batches. Setting
                max_batch_size < num_seeds will slow down optimization.
            log:
                If True, returns intermediate optimization results.
            remove_cache:
                Removes cached stuff from auto_detect()
            st_progress_bar:
                If true, uses streamlit progress bar

        Returns
            variables
                Variable object. Check variable_manager to see how it is
                formatted.
            outs
                The output of the optimization.
            losses
                If log=True, returns intermediate losses.
        """

        # -- Check if auto_detection was run -- #
        if cls_lbl is None:
            if hasattr(self, '_cls_lbl'):
                print('Found class label from auto_detect()')
                cls_lbl = self._cls_lbl
                if remove_cache:
                    del self._cls_lbl # For safety reasons

        if mask is None:
            if hasattr(self, '_mask'):
                print('Found mask from auto_detect()')
                mask = self._mask
                if remove_cache:
                    del self._mask # For safety reasons


        # -- Prepare variables -- #
        im, cv, mask = self.prepare_input(im, cls_lbl, mask)
        var_manager = VariableManager(lr=0.05,
                                      cv_lr=1e-4,
                                      precision='float',
                                      cv_search_method='grad',
                                      optimize_t=False)


        # -- Transformation -- #
        transform_fn = TF.ComposeTransform([
            (TF.SpatialTransform(optimize=True), 1.0),
            (TF.BrightnessTransform(), 3.0)
        ])

        t = transform_fn.get_param()
        t[0] = compute_pre_alignment(binarize(mask))
        t = torch.from_numpy(np.concatenate(t)).unsqueeze(0).float()

        var_manager.set_default(num_seeds=num_seeds, cv=cv, t=t,
                                target=im, weight=mask)


        # -- Optimizer -- #
        if num_seeds == 18:
            opt = optimizers.BasinCMAOptimizer(
                                        model=self.model,
                                        max_batch_size=max_batch_size,
                                        log=log,
                                        )
        else:
            print('Number of seed is set below 18, using Nevergrad CMA.')
            # Note:
            # This performs worse than CMA implementation above.
            opt = optimizers.NevergradHybridOptimizer(
                                        method='CMA',
                                        model=self.model,
                                        max_batch_size=max_batch_size,
                                        log=log,
                                        )

        opt.register_transform_fn(transform_fn)
        opt.register_loss_fn(self.loss_fn)


        # -- Search transformation -- #
        print('Searching for transformation')
        if st_progress_bar:
            import streamlit as st
            st.sidebar.markdown('<h4> Optimizing transformation </h4>',
                                unsafe_allow_html=True)
            st_transform_pbar = st.sidebar.progress(0)

            st.sidebar.markdown('<h4> Projecting image </h4>',
                                unsafe_allow_html=True)
            st_project_pbar = st.sidebar.progress(0)
        else:
            st_transform_pbar = None
            st_project_pbar = None

        _var_manager = copy.deepcopy(var_manager)
        t, _, _ = search_transform(
                               model=self.model,
                               transform_fn=transform_fn,
                               var_manager=_var_manager,
                               loss_fn=self.loss_fn,
                               meta_steps=transform_adam_steps,
                               grad_steps=transform_cma_steps,
                               pbar=st_transform_pbar,
                               log=log,
                               )
        var_manager.set_default(t=t)


        # -- Encoder -- #
        z_init = None
        if encoder_init:
            if not hasattr(self, 'biggan_encoder'):
                self.biggan_encoder = load_biggan_encoder()

            print('Warm starting with an Encoder')
            with torch.no_grad():
                z_init = self.biggan_encoder(im, cv)
            var_manager.set_default(z=(z_init, 0.5))
            z_init = (to_numpy(z_init[0]), 0.5)


        # -- Optimize -- #
        variables, outs, losses = opt.optimize(
                            var_manager, cma_steps, adam_steps, finetune_steps,
                            128, z_init, pbar=st_project_pbar)

        return variables, outs, losses, transform_fn


    def auto_detect(self, im, mask_type='segmentation'):
        if not hasattr(self, 'detector'):
            self.detector = Detector()

        if not hasattr(self, 'classifier'):
            self.classifier = Classifier()

        candidates = self.detector(im, is_tensor=False)

        if candidates == None:
            print('Did not find any valid object in the image.')

        else:
            det_bboxes = candidates['boxes']
            det_labels = candidates['labels']
            det_scores = candidates['scores']
            det_masks = candidates['masks']

            coco_to_wnid = imagenet_tools.get_coco_valid_wnids()

            # Start from highest to lowest score
            for idx in np.argsort(det_scores.cpu().numpy())[::-1]:
                det_cls_noun = COCO_INSTANCE_CATEGORY_NAMES[det_labels[idx]]
                bbox = det_bboxes[idx].cpu().numpy().astype(np.int)

                bbox_im = im[bbox[1]: bbox[3], bbox[0]: bbox[2], :]

                top5_cls = self.classifier(bbox_im, is_tensor=False, top5=True)
                pred_cls = top5_cls[1][0].item()

                misc = []
                for c in top5_cls[1]:
                    misc.append([c.item(), IMAGENET_LABEL_TO_NOUN[c.item()]])

                pred_wnid = IMAGENET_LABEL_TO_WNID[pred_cls]
                pred_cls_noun = IMAGENET_LABEL_TO_NOUN[pred_cls]

                valid_wnids = coco_to_wnid[det_cls_noun]

                if pred_wnid in valid_wnids:
                    print(('Found a match. Classified class {} is in the ' +
                           'detected class {}').format(
                           pred_cls_noun, det_cls_noun))

                    if mask_type == 'segmentation':
                        m = det_masks[idx] > 0.5

                    elif mask_type == 'bbox':
                        m = torch.zeros(1, im.shape[0], im.shape[1])
                        m[:, bbox[1]: bbox[3], bbox[0]: bbox[2]] = 1.0

                    else:
                        invalid_msg = 'Invalid mask_type {}'
                        raise ValueError(invalid_msg.format(mask_type))

                    self._cls_lbl = pred_cls
                    m = to_image(m, denormalize=False)
                    m = cv2.medianBlur(m.astype(np.uint8), 5)
                    self._mask = m.reshape(m.shape[0], m.shape[1], 1) / 255.
                    return self._mask, pred_cls_noun, det_cls_noun, misc

                print(('Classification and Detection is inconsistent. ' +
                       'Classified class {} is not an element of the ' +
                       'detected class {}. Trying next candidate').format(
                       pred_cls_noun, det_cls_noun))

        print('Auto-detection failed. All candidates are invalid.')

        cls_lbl = self.classifier(im, as_onehot=False, is_tensor=False)
        self._cls_lbl = cls_lbl
        print('Mask is set to None and the predicted class is: {} ({})'.format(
                cls_lbl, IMAGENET_LABEL_TO_NOUN[cls_lbl]))
        return



if __name__ == '__main__':
    import argparse
    from im_utils import smart_resize, center_crop, poisson_blend

    parser = argparse.ArgumentParser()
    parser.add_argument('--im', type=str,
                        default='./examples/very-cute-doggo.jpg',
                        help='path to an image')
    parser.add_argument('--mask', type=str,
                        help='path to mask image')
    parser.add_argument('--cls', type=int,
                        help='ImageNet imagenet class label')
    parser.add_argument('--encoder_init', action='store_true',
                        help='uses encoder to initialize search')
    args = parser.parse_args()

    assert os.path.exists(args.im)
    fn = os.path.basename(args.im)

    im = cv2.imread(args.im)[:, :, [2, 1, 0]]
    solver = TransformableBasinCMAProjection()
    im = smart_resize(center_crop(im))

    if args.mask:
        mask = cv2.imread(args.mask)[:, :, :1]
    else:
        solver.auto_detect(im, mask_type='bbox')
        mask = solver._mask

    if args.cls:
        cls_lbl = args.cls
    else:
        if not hasattr(solver, '_cls_lbl'):
            solver.auto_detect(im, mask_type='bbox')
        cls_lbl = solver._cls_lbl

    variables, outs, losses, transform_fn = \
                            solver(im, mask=mask, cls_lbl=cls_lbl,
                                   encoder_init=args.encoder_init,
                                   log=False)

    idx = np.argmin(losses).squeeze()
    z = torch.stack(variables.z.data)
    cv = torch.stack(variables.cv.data)

    with torch.no_grad():
        out = solver.model(z=z[idx:idx+1], c=cv[idx:idx+1])
        out_im = to_image(out)[0]

    t = torch.stack(variables.t.data)[idx:idx+1]

    inv_im = to_image(transform_fn(out.cpu(), t.cpu(), invert=True))[0]
    blended = poisson_blend(im[:, :, [2, 1, 0]], mask, inv_im)

    os.makedirs('./results', exist_ok=True)

    jpg_quality = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    cv2.imwrite('./results/projected-{}.jpg'.format(fn), inv_im, jpg_quality)
    cv2.imwrite('./results/blended-{}.jpg'.format(fn), blended, jpg_quality)
