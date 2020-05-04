import init_paths
import nevergrad as ng

import numpy as np
from optimization_core import step
from cma_optimizer import CMA
from im_utils import binarize, to_image, make_grid
from variable_manager import override_variables
from misc import to_numpy, progress_print, cprint
import torch


# Base classes for Optimizers

class BaseOptimizer():
    """ Base template for gradient optimization """

    def __init__(self, model, log=True, log_iter=10, max_batch_size=9,
                 *args, **kwargs):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.log = log
        self.log_iter = log_iter
        self.show_iter = 50
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


class BaseCMAOptimizer():
    """
    Base template for CMA optimization. Should be used jointly with
    BaseOptimizer.
    """

    def __init__(self):
        """
        Attribute:
            inverted_loss
                If inverted_loss is True, compute the loss on the original
                target image. If False, use the last iterate loss. Setting
                this to True may result in better result but slightly slower
                run-time.
        """
        super().__init__()
        self.inverted_loss = True
        return

    def setup_cma(self, cma_dim=128, cma_init=None):
        """
        Args
            cma_dim
                Dimension of the CMA optimizer
            cma_init
                cma_init can be a list or a tuple. If tuple (mu, sigma).
                If cma_init is only mu, sigma is set to 1.0
        """
        if cma_init:
            if type(cma_init) == tuple:
                assert len(cma_init[0]) == cma_dim
                self.cma_opt = CMA(*cma_init)
            else:
                assert len(cma_init) == cma_dim
                self.cma_opt = CMA(cma_init)
        else:
            self.cma_opt = CMA([0] * cma_dim)
        return

    def cma_init(self, var_manager, cma_var='z', is_last_iter=False):
        if is_last_iter:
            ns = var_manager.num_seeds
            variables = var_manager.init(num_seeds=ns)
            override_variables(variables, [[cma_var, self.cma_opt.ask(ns)]])
        else:
            variables = var_manager.init(num_seeds=self.cma_opt.batch_size())
            override_variables(variables, [[cma_var, self.cma_opt.ask()]])
        return variables

    def cma_update(self, v, cma_var='z'):
        cma_v = to_numpy(torch.stack(v[cma_var].data))

        if self.inverted_loss:
            with torch.no_grad():
                z, cv = torch.stack(v.z.data), torch.stack(v.cv.data)
                out = self.model(z=z, c=cv)

                if v.t is not None:
                    t = torch.stack(v.t.data)
                    out = self.transform_fn(out, t, invert=True)

                mask = binarize(v.weight.data[0]).unsqueeze(0)
                target = v.target.data[0].unsqueeze(0)
                loss = self.loss_fn(out, target, mask).detach().cpu().numpy()
                self.cma_opt.tell(cma_v, loss)
        else:
            self.cma_opt.tell(cma_v, self.loss)
        return


# Useable Optimizers


class GradientOptimizer(BaseOptimizer):
    """
    Basic gradient optimizer. Compatible with any gradient-based optimizer.
    The optimiation method defined in variable_manager is used.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def optimize(self, var_manager, grad_steps, pbar=None):
        """
        Args
            var_manager:
                variable manager for variable creation.
            grad_steps:
                number of gradient descent updates.
            pbar:
                progress bar such as tqdm or st.progress
        """
        self.losses, self.outs = [], []

        variables = var_manager.init()
        for i in range(grad_steps):
            self.step(variables, optimize=True)

            if pbar is not None:
                pbar.progress(i / total_steps)

            if self.log:
                if ((i + 1) % self.log_iter == 0) or (i + 1 == grad_steps):
                    self.log_result(variables, i + 1)

            if (i + 1) % self.show_iter == 0:
                progress_print('optimize', i + 1, grad_steps, 'c')

        if self.log:
            return variables, self.outs, self.losses
        return variables, self.out, self.loss


class CMAOptimizer(BaseOptimizer, BaseCMAOptimizer):
    """
    CMA optimizer. Gradient descent can be used to further optimize the seeds
    from CMA.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def optimize(self, var_manager, meta_steps, grad_steps=0, cma_dim=128,
                 cma_init=None, pbar=None):
        """
        Args
            var_manager:
                variable manager for variable creation.
            grad_steps:
                number of gradient descent updates.
            meta_steps:
                number of CMA updates
            grad_steps:
                number of gradient updates to apply after CMA optimization
            cma_dim:
                dimension of CMA
            cma_init:
                initialize cma with the provided mean (optional: mean and sigma)
            pbar:
                progress bar such as tqdm or st.progress
        """
        self.setup_cma(cma_dim, cma_init)
        self.losses, self.outs, i = [], [], 0
        total_steps = meta_steps + grad_steps

        # CMA optimization
        for _ in range(meta_steps):
            variables = self.cma_init(var_manager, 'z', is_last_iter=False)
            self.step(variables, optimize=False)
            i += 1

            if self.log:
                if ((i + 1) % self.log_iter == 0) or (i + 1 == grad_steps):
                    self.log_result(variables, i + 1)

            self.cma_update(variables, 'z')

            if pbar is not None:
                pbar.progress(i / total_steps)
            else:
                if (i + 1) % self.show_iter == 0:
                    progress_print('optimize', i + 1, total_steps, 'c')

        # Finetune CMA with ADAM optimization
        variables = self.cma_init(var_manager, 'z', is_last_iter=True)

        for _ in range(grad_steps):
            self.step(variables, optimize=True)
            i += 1

            if self.log:
                if ((i + 1) % self.log_iter == 0) or (i + 1 == grad_steps):
                    self.log_result(variables, i + 1)

            if pbar is not None:
                pbar.progress(i / total_steps)
            else:
                if (i + 1) % self.show_iter == 0:
                    progress_print('optimize', i + 1, total_steps, 'c')

        if self.log:
            return variables, self.outs, self.losses
        return variables, self.out, self.loss


class BasinCMAOptimizer(BaseOptimizer, BaseCMAOptimizer):
    """
    Optimize using BasinCMA. BasinCMA interleaves CMA updates with ADAM updates.
    """

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def optimize(self, var_manager, meta_steps, grad_steps,
                 finetune_grad_steps=300, cma_dim=128, cma_init=None,
                 pbar=None):
        """
        Args
            var_manager:
                variable manager for variable creation.
            grad_steps:
                number of gradient descent updates.
            meta_steps:
                number of CMA updates
            grad_steps:
                number of gradient updates per CMA update.
            finetune_grad_steps:
                after the final iteration of BasinCMA, further optimize the
                last drawn samples using gradient descent.
            cma_dim:
                dimension of CMA
            cma_init:
                initialize cma with the provided mean (optional: mean and sigma)
            pbar:
                progress bar such as tqdm or st.progress
        """
        self.setup_cma(cma_dim, cma_init)
        self.losses, self.outs, i = [], [], 0
        total_steps = meta_steps * grad_steps + finetune_grad_steps

        for meta_iter in range(meta_steps + 1):
            is_last_iter = (meta_iter == meta_steps)
            _grad_steps = finetune_grad_steps if is_last_iter else grad_steps

            variables = self.cma_init(var_manager, 'z', is_last_iter)

            for _ in range(_grad_steps):
                self.step(variables, optimize=True)
                i += 1

                if self.log:
                    if ((i + 1) % self.log_iter == 0) or (i + 1 == grad_steps):
                        self.log_result(variables, i + 1)

                if pbar is not None:
                    pbar.progress(i / total_steps)
                else:
                    if (i + 1) % self.show_iter == 0:
                        progress_print('optimize', i + 1, total_steps, 'c')

            if not is_last_iter:
                self.cma_update(variables, 'z')

        if self.log:
            return variables, self.outs, self.losses
        return variables, self.out, self.loss


# ---  Experimental code below --- ##


class NevergradOptimizer(BaseOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(method, *args, **kwargs)
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

    def optimize(self, var_manager, meta_steps, grad_steps=300,
                 meta_dim=128, meta_init=None, pbar=None):
        """
        Args
            var_manager:
                variable manager for variable creation.
            meta_steps:
                number of outer-loop updates. This is the number of updates for
                the nevergrad optimizer.
            grad_steps:
                number of gradient updates to apply after gradient-free
                optimization.
            meta_dim:
                dimension of the variable to optimize using gradient-free
                optimizer.
            meta_init:
                initialize nevergrad optimizer with the provided mean
                (optional: mean and sigma)
            pbar:
                progress bar such as tqdm or st.progress
        """
        # -- Setup optimizer -- #
        opt_fn = ng.optimizers.registry[self.method]
        if meta_init is None:
            p = ng.p.Array(shape=(meta_dim,)).set_mutation(sigma=1.0)
            opt = opt_fn(parametrization=p, budget=meta_steps + 1)
        else:
            p = ng.p.Array(meta_init[0]).set_mutation(sigma=meta_init[1])
            opt = opt_fn(parametrization=p, budget=meta_steps + 1)

        total_steps = meta_steps + grad_steps
        self.losses, self.outs, i = [], [], 0

        # -- Start optimization -- #
        for _ in range(meta_steps):
            if self.is_sequential:
                variables = var_manager.init(num_seeds=1)
                _z = opt.ask()
                z = np.array(_z.args)
            else:
                variables = var_manager.init()
                _z = [opt.ask() for _ in range(var_manager.num_seeds)]
                z = np.concatenate([np.array(x.args) for x in _z])

            override_variables(variables, [['z', z]])  # 18 seeds
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

            if pbar is not None:
                pbar.progress(i / total_steps)
            else:
                if (i + 1) % self.show_iter == 0:
                    progress_print('optimize', i + 1, meta_steps, 'c')

        # -- Finetune the final seeds with ADAM -- #
        variables = var_manager.init()
        _z = [opt.ask() for _ in range(var_manager.num_seeds)]
        z = np.concatenate([np.array(x.args) for x in _z])
        override_variables(variables, [['z', z]])

        for _ in range(grad_steps):
            self.step(variables, optimize=True)
            i += 1

            if self.log:
                if ((i + 1) % self.log_iter == 0) or (i + 1 == grad_steps):
                    self.log_result(variables, i + 1)

            if pbar is not None:
                pbar.progress(i / total_steps)
            else:
                if (i + 1) % self.show_iter == 0:
                    progress_print('optimize', i + 1, total_steps, 'c')

        if self.log:
            return variables, self.outs, self.losses
        return variables, self.out, self.loss


class NevergradHybridOptimizer(BaseOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def optimize(self, var_manager, meta_steps, grad_steps,
                 finetune_grad_steps=300, meta_dim=128, meta_init=None,
                 pbar=None):
        """
        Args
            var_manager:
                variable manager for variable creation.
            meta_steps:
                number of outer-loop updates. This is the number of updates for
                the nevergrad optimizer.
            grad_steps:
                number of gradient updates per meta_step.
            finetune_grad_steps:
                after the final iteration of nevergrad, further optimize the
                last drawn samples using gradient descent.
            meta_dim:
                dimension of the variable to optimize using gradient-free
                optimizer.
            meta_init:
                initialize nevergrad optimizer with the provided mean
                (optional: mean and sigma)
            pbar:
                progress bar such as tqdm or st.progress
        """

        # -- Setup optimizer -- #
        opt_fn = ng.optimizers.registry[self.method]
        if meta_init is None:
            p = ng.p.Array(shape=(meta_dim,)).set_mutation(sigma=1.0)
            opt = opt_fn(parametrization=p, budget=meta_steps + 1)
        else:
            p = ng.p.Array(init=meta_init[0]).set_mutation(sigma=meta_init[1])
            opt = opt_fn(parametrization=p, budget=meta_steps + 1)

        self.losses, self.outs, i = [], [], 0
        total_steps = meta_steps * grad_steps + finetune_grad_steps

        # -- Start optimization -- #
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

                if pbar is not None:
                    pbar.progress(i / total_steps)

                if self.log:
                    if ((i + 1) % self.log_iter == 0) or (i + 1 == grad_steps):
                        self.log_result(variables, i + 1)

                if pbar is not None:
                    pbar.progress(i / total_steps)
                else:
                    if (i + 1) % self.show_iter == 0:
                        progress_print('optimize', i + 1, total_steps, 'c')

            if self.is_sequential:
                opt.tell(_z, self.loss[0])
            else:
                for z, l in zip(_z, self.loss):
                    opt.tell(z, l)

        if self.log:
            return variables, self.outs, self.losses
        return variables, self.out, self.loss
