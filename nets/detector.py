import torch
from torchvision import models, transforms
from imagenet_tools import COCO_INSTANCE_CATEGORY_NAMES
import numpy as np


class Detector():
    def __init__(self):
        self.model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.float().cuda().eval()
        # maskrcnn expects [0, 1] range
        self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            ])
        return

    def prep(self, im, is_tensor=True):
        if is_tensor:
            im = ((im + 1.0) / 2.0)
        else:
            im = self.transform(im).unsqueeze(0)
        return im.float().cuda()

    def __call__(self, x, find=None, is_tensor=True):
        with torch.no_grad():
            x = self.prep(x, is_tensor=is_tensor)
            x = self.model(x)[0]

            # The output is a dictionary consisting:
            # 'boxes', 'labels', 'scores', 'masks'

            if find is None:
                return x

            find_lbl = np.argwhere(np.array(COCO_INSTANCE_CATEGORY_NAMES) == find)
            find_lbl = np.squeeze(find_lbl)
            for bbox, lbl, score in zip(x['boxes'], x['labels'], x['scores']):
                if lbl.item() == find_lbl:
                    return lbl, bbox, score
            return None
