from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn
import loss_functions as LF
from types import LambdaType


class Benchmark():
    def __init__(self, metrics):
        """
        Computes benchmark while optimizing.
        Args:
            metrics:
                List of metrics to compute. For perceptual metrics, the
                model is only initialized after evaluate has been called.
        """
        valid_metrics = {'l1':LF.ReconstructionLoss(loss_type='l1'),
                         'l2':LF.ReconstructionLoss(loss_type='l2'),
                         'alex':lambda: LF.PerceptualLoss('alex'),
                         'squeeze':lambda: LF.PerceptualLoss('squeeze'),
                         'vgg':lambda: LF.PerceptualLoss('vgg')}

        self.metrics = {}
        for m in metrics:
            if m in valid_metrics.keys():
                self.metrics[m] = valid_metrics[m]
            else:
                raise ValueError('Invalid metric {}'.format(m))
        return

    def evaluate(self, out, target, mask):
        result = {}
        with torch.no_grad():
            out = out.cuda()
            for metric, metric_fn in self.metrics.items():
                if type(metric_fn) is LambdaType:
                    # initialize it if not initialized
                    self.metrics[metric] = metric_fn()
                    metric_fn = self.metrics[metric]
                loss = metric_fn(out, target, mask).detach().cpu().numpy()
                result[metric] = loss
        return result
