import init_paths
import nevergrad as ng

import numpy as np
from optimization_core import step
from cma_optimizer import CMA
from im_utils import binarize, to_image, make_grid
from variable_manager import override_variables
from misc import to_numpy, progress_print, cprint
import torch
import numpy as np



class BaseOptimizer():
    def __init__(self, model, log=True, log_iter=10, max_batch_size=9, *args):
        self.max_batch_size = max_batch_size
        self.log = log
        self.log_iter = log_iter
        self.model = model
        self.transform_fn = None
        return

    def register_benchmark(self, benchmark):
        self.bm = benchmark
        return

    def register_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn
        return

    def register_transform_fn(self, transform_fn):
        self.transform_fn = transform_fn
        return

    def step(self, variables, optimize=True):
        self.out, self.loss, self.other = step(
                                            self.model, variables,
                                            loss_fn=self.loss_fn,
                                            transform_fn=self.transform_fn,
                                            optimize=optimize,
                                            max_batch_size=self.max_batch_size
                                            )
        return self.out, self.loss, self.other

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
            self.losses.append([step_iter, res])
        else:
            res = {'loss': np.array(self.loss)}
        self.outs.append(to_image(make_grid(self.out), cv2_format=False))
        return


class GradientOptimizer(BaseOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        return

    def optimize(self, var_manager, grad_steps):
        self.losses, self.outs = [], []

        variables = var_manager.init()
        for i in range(grad_steps):
            self.step(variables, optimize=True)

            if self.log:
                if ((i + 1) % self.log_iter == 0) or (i + 1 == grad_steps):
                    self.log_result(variables, i + 1)

            if (i + 1) % 50 == 0:
                progress_print('optimize', i + 1, grad_steps, 'c')

        if self.log:
            return variables, self.outs, self.losses
        return variables, self.out, self.loss


class CMAOptimizer(BaseOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        return

    def optimize(self, var_manager, meta_steps, finetune_grad_steps=0,
                 cma_z_init=None):
        cma_opt = CMA(*cma_z_init) if cma_z_init is not None else CMA()
        grad_steps = finetune_grad_steps # for code compactness
        total_steps = meta_steps + finetune_grad_steps
        self.losses, self.outs, i = [], [], 0

        # CMA optimization
        for _ in range(meta_steps):
            variables = var_manager.init(num_seeds=cma_opt.batch_size())

            override_variables(variables, [['z', cma_opt.ask()]]) # 18 seeds
            self.step(variables, optimize=False)
            i += 1

            if self.log:
                if ((i + 1) % self.log_iter == 0) or (i + 1 == grad_steps):
                    self.log_result(variables, i + 1)

            cma_opt.tell(to_numpy(torch.stack(variables.z.data)), self.loss)

            if (i + 1) % 50 == 0:
                progress_print('optimize', i + 1, total_steps, 'c')

        # Finetune CMA with ADAM optimization
        variables = var_manager.init()
        override_variables(
                variables, [['z', cma_opt.ask(var_manager.num_seeds)]])

        for _ in range(grad_steps):
            self.step(variables, optimize=True)
            i += 1

            if self.log:
                if ((i + 1) % self.log_iter == 0) or (i + 1 == grad_steps):
                    self.log_result(variables, i + 1)

            if (i + 1) % 50 == 0:
                progress_print('optimize', i + 1, total_steps, 'c')

        if self.log:
            return variables, self.outs, self.losses
        return variables, self.out, self.loss


class BasinCMAOptimizer(BaseOptimizer):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def optimize(self, var_manager, meta_steps, grad_steps, cma_z_init=None,
                 finetune_grad_steps=300):
        cma_opt = CMA(*cma_z_init) if cma_z_init is not None else CMA()

        self.losses, self.outs, i = [], [], 0
        total_steps = meta_steps * grad_steps + finetune_grad_steps

        for meta_iter in range(meta_steps + 1):
            is_last_iter = (meta_iter == meta_steps)

            if is_last_iter:
                ns = var_manager.num_seeds
                variables = var_manager.init(num_seeds=ns)
                override_variables(variables, [['z', cma_opt.ask(ns)]])
            else:
                variables = var_manager.init(num_seeds=cma_opt.batch_size())
                override_variables(variables, [['z', cma_opt.ask()]])

            _grad_steps = finetune_grad_steps if is_last_iter else grad_steps

            for _ in range(_grad_steps):
                self.step(variables, optimize=True)
                i += 1

                if self.log:
                    if ((i + 1) % self.log_iter == 0) or (i + 1 == grad_steps):
                        self.log_result(variables, i + 1)

                if (i + 1) % 50 == 0:
                    progress_print('optimize', i + 1, total_steps, 'c')

            if not is_last_iter:
                cma_opt.tell(to_numpy(torch.stack(variables.z.data)), self.loss)

        if self.log:
            return variables, self.outs, self.losses
        return variables, self.out, self.loss


class NevergradOptimizer(BaseOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.method = kwargs['method']
        self.valid_methods = [x[0] for x in ng.optimizers.registry.items()]
        self.sequential_methods = ['SQPCMA', 'chainCMAPowell', 'Powell']
        self.is_sequential = self.method in self.sequential_methods
        if self.is_sequential:
            seq_msg = '{} is a sequential method. batch size is set to 1'
            cprint(seq_msg.format(self.method), 'y')
        assert self.method in self.valid_methods, \
                'unknown nevergrad method: {}'.format(self.method)
        return

    def optimize(self, var_manager, meta_steps, finetune_grad_steps=0,
                 cma_z_init=None):
        p = ng.p.Array(shape=(128,)).set_mutation(sigma=1)
        opt_fn = ng.optimizers.registry[self.method]
        opt = opt_fn(parametrization=p, budget=meta_steps)

        self.losses, self.outs, i = [], [], 0

        for _ in range(meta_steps):
            if self.is_sequential:
                variables = var_manager.init(num_seeds=1)
                _z = opt.ask()
                z = np.array(_z.args)
            else:
                variables = var_manager.init()
                _z = [opt.ask() for _ in range(var_manager.num_seeds)]
                z = np.concatenate([np.array(x.args) for x in _z])

            override_variables(variables, [['z', z]]) # 18 seeds
            self.step(variables, optimize=False)
            i += 1

            if self.log:
                if ((i + 1) % self.log_iter == 0) or (i + 1 == grad_steps):
                    self.log_result(variables, i + 1)

            if self.is_sequential:
                opt.tell(_z, self.loss[0])
            else:
                for x, l in zip(_x, self.loss):
                    opt.tell(z, l)

            if (i + 1) % 50 == 0:
                progress_print('optimize', i + 1, meta_steps, 'c')

        if self.log:
            return variables, self.outs, self.losses
        return variables, self.out, self.loss


class NevergradHybridOptimizer(BaseOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.method = kwargs['method']
        self.valid_methods = [x[0] for x in ng.optimizers.registry.items()]
        self.sequential_methods = ['SQPCMA', 'chainCMAPowell', 'Powell']
        self.is_sequential = self.method in self.sequential_methods
        if self.is_sequential:
            seq_msg = '{} is a sequential method. batch size is set to 1'
            cprint(seq_msg.format(self.method), 'y')
        assert self.method in self.valid_methods, \
                'unknown nevergrad method: {}'.format(self.method)
        return

    def optimize(self, var_manager, meta_steps, grad_steps, cma_z_init=None,
                 finetune_grad_steps=300):

        p = ng.p.Array(shape=(128,)).set_mutation(sigma=1)
        opt_fn = ng.optimizers.registry[self.method]
        opt = opt_fn(parametrization=p, budget=meta_steps)

        self.losses, self.outs, i = [], [], 0
        total_steps = (meta_steps - 1) * grad_steps + finetune_grad_steps

        for meta_iter in range(meta_steps + 1):

            if self.is_sequential:
                variables = var_manager.init(num_seeds=1)
                _z = opt.ask()
                z = np.array(_z.args)
            else:
                variables = var_manager.init()
                _z = [opt.ask() for _ in range(var_manager.num_seeds)]
                z = np.concatenate([np.array(x.args) for x in _z])

            override_variables(variables, [['z', z]])

            is_last_iter = (meta_iter == meta_steps)
            _grad_steps = finetune_grad_steps if is_last_iter else grad_steps

            for _ in range(_grad_steps):
                self.step(variables, optimize=True)
                i += 1

                if self.log:
                    if ((i + 1) % self.log_iter == 0) or (i + 1 == grad_steps):
                        self.log_result(variables, i + 1)

                if (i + 1) % 50 == 0:
                    progress_print('optimize', i + 1, total_steps, 'c')

            if self.is_sequential:
                opt.tell(_z, self.loss[0])
            else:
                for z, l in zip(_z, self.loss):
                    opt.tell(z, l)

        if self.log:
            return variables, self.outs, self.losses
        return variables, self.out, self.loss
