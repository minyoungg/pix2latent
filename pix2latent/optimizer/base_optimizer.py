import torch
import numpy as np
import cv2

from pix2latent.optimizer.closure import step
from pix2latent.utils.image import to_image, to_grid


class _BaseOptimizer():
    """ Base template for gradient optimization """

    def __init__(self, model, var_manager, loss_fn, max_batch_size=9,
                 log=False, track_variables=True, **kwargs):
        """
        Args
            model (nn.Module): a model to invert
            var_manager (VariableManager): instance of the variable manager
            loss_fn (lambda function): loss function to compute gradients with
            max_batch_size (int): maximum batch size. if number of seeds
                exceeds the maximum batch size, it will be divided into
                multiple mini-batches

        """
        self.max_batch_size = max_batch_size
        self.model = model.eval()
        self.var_manager = var_manager
        self.loss_fn = loss_fn
        self.transform_fns = {}

        self.log = log
        self.log_iter = 5
        self.show_iter = 50
        self.log_resize_factor = None
        self.track_variables = track_variables
        self.tracked = {}
        return


    def register_benchmark(self, benchmark):
        self.bm = benchmark
        return


    def register_transform(self,
                           transform_fn,
                           tranform_var_name,
                           target_var_name):
        """
        Applies transformation function using the transform_var on the target
        variables before optimizing.
        """

        self.transform_fns[target_var_name] = {
                                'fn': transform_fn,
                                'transform_param': tranform_var_name,
                                'target_var': target_var_name
                                }

        return


    def apply_transform(self, variables, transform_dict):
        t_fn = transform_dict['fn']
        src_name = transform_dict['transform_param']
        dst_name = transform_dict['target_var']

        src_type = self.var_manager.variable_info[src_name]['var_type']
        dst_type = self.var_manager.variable_info[dst_name]['var_type']

        src_data = torch.stack(variables[src_type][src_name].data)
        dst_data = torch.stack(variables[dst_type][dst_name].data)

        new_dst_data = list(t_fn(dst_data, src_data))

        for i in range(len(new_dst_data)):
            variables[dst_type][dst_name].data[i].data = new_dst_data[i].data

        return


    def step(self, variables, optimize=True, transform=False):
        if len(self.transform_fns) > 0 and transform:

            for _, transform_dict in self.transform_fns.items():
                self.apply_transform(variables, transform_dict)

        if self.track_variables:
            self.track(variables)

        self.out, self.loss, self.other = step(
                    self.model, variables,
                    loss_fn=self.loss_fn,
                    optimize=optimize,
                    max_batch_size=self.max_batch_size
        )

        return self.out, self.loss, self.other


    def track(self, variables):
        for v_name, v_data in variables.input.items():
            if v_name not in self.tracked.keys():
                self.tracked[v_name] = []

            self.tracked[v_name] += \
                        [torch.stack(v_data.data).cpu().detach().clone()]
        return


    def optimize(self):
        raise NotImplementedError


    def benchmark(self, variables, out):
        if variables.t is not None:
            out = self.transform_fn(
                out, torch.stack(variables.t.data), invert=True)
        res = self.bm.evaluate(out, variables.target.data[0].unsqueeze(0),
                               binarize(variables.weight.data[0]).unsqueeze(0))
        return res


    def log_result(self, variables, step_iter):
        if hasattr(self, 'bm'):
            res = self.benchmark(variables, self.out)
        else:
            res = {'loss': np.array(self.loss)}
        self.losses.append([step_iter, res])

        collage = to_image(to_grid(self.out.cpu()), cv2_format=False)

        if self.log_resize_factor is not None:
            collage = cv2.resize(
                            np.array(collage, dtype=np.uint8), None,
                            fx=self.log_resize_factor,
                            fy=self.log_resize_factor,
                            interpolation=cv2.INTER_AREA,
                            )

        self.outs.append(collage)
        return
