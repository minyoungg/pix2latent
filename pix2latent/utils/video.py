from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
import os
import imageio
import skvideo.io



def make_gif(save_path, ims, duration=20.0):
    """ dumps a list of images into gif """
    dpf = duration / len(ims)  # limited to 12fps
    imageio.mimsave(save_path, ims, duration=dpf)
    return


def make_video(save_path, ims, fps=30, duration=None, safe=True):
    """
    Creates a video given an array of images. Uses FFMPEG backend.
    Depending on the FFMPEG codec supported you might have to change pix_fmt
    To change the quality of the saved videos, change -b (the encoding bitrate)
    Args:
      save_path: path to save the video
      ims: an array of images
      fps: frames per seconds to save the video
      duration: the duration of the video, if not None, will override fps.

    Examples:
        >>> ims = [im1, im2, im3]
        >>> make_video('video.webm', ims, fps=10)
    """

    ims = np.array(ims)

    if np.max(ims) <= 1:
        ims = (ims * 255)

    ims = ims.astype(np.uint8)

    if duration is not None:
        fps = len(ims) / duration

    height, width = ims[0].shape[:2]

    if save_path.endswith('webm'):
        fourcc = cv2.VideoWriter_fourcc(*'VP90')
        video = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        for im in ims:
            video.write(im[:, :, [2,1,0]])

        video.release()

    elif save_path.endswith('mp4'):
        skvideo.io.vwrite(save_path, ims,
                      inputdict={'-r': str(fps)},
                      outputdict={'-r': str(fps),
                                  '-pix_fmt': 'yuv420p',
                                  '-b': '40000000'})

    else:
        print('unsupported video format')
        return False

    print('saved video to {}'.format(save_path))
    return
