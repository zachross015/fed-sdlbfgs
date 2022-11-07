import torch
from functools import reduce
from torch.optim import Optimizer
from collections import OrderedDict

class SdLBFGSLayer(Optimizer):
    """PyTorch Implementation of SdLBFGS algorithm [1].

    Code is adopted from LBFGS in PyTorch and modified by
    Huidong Liu (h.d.liew@gmail.com) and Yingkai Li
    (yingkaili2023@u.northwestern.edu). Further modification to include Delta
    done by Zachary Ross (zlr80401@uga.edu).

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Arguments:
        lr (float): learning rate (default: 1).
        lr_decay (bool): whether to perform learning rate decay (default: True).
        weight_decay (float): weight decay (default: 0).
        max_iter (int): maximal number of iterations per optimization step
            (default: 1).
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).

    [1] Wang, Xiao, et al. "Stochastic quasi-Newton methods for nonconvex stochastic optimization."
    SIAM Journal on Optimization 27.2 (2017): 927-956.
    """

    def __init__(self, params, lr=0.1, lr_decay=True, weight_decay=0, max_iter=1, max_eval=None, tolerance_grad=1e-5, tolerance_change=1e-9, history_size=100, delta=0.1):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, max_iter=max_iter,
                        max_eval=max_eval,
                        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                        history_size=history_size, delta=delta)
        super(SdLBFGSLayer, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("SdLBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

        self.state['global_state']['local_state'] = []
        for i, p in enumerate(self._params): 
            self.state['global_state']['local_state'].append(OrderedDict())
            self.state['global_state']['local_state'][i].setdefault('func_evals', 0)
            self.state['global_state']['local_state'][i].setdefault('n_iter', 0)


    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_grads(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return views

    def _add_grad(self, updates):
        for j, p in zip(updates, self._params):
            step_size, update = j
            p.data.add_(step_size, update.view_as(p.data))

    def _add_weight_decay(self, weight_decay, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            update[offset:offset + numel].add_(weight_decay, p.data.view(-1))
            offset += numel
        return update

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        lr = group['lr']
        lr_decay = group['lr_decay']
        weight_decay = group['weight_decay']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        history_size = group['history_size']
        delta = group['delta']

        state = self.state['global_state']
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        # evaluate initial f(x) and df/dx
        loss = None
        if closure is not None:
            loss = closure().data[0]

        current_evals = 1
        state['func_evals'] += 1

        grads = self._gather_grads()

        updates = []

        for i in range(len(grads)):

            flat_grad = grads[i]
            local_state = state['local_state'][i]
            abs_grad_sum = flat_grad.abs().sum()

            if abs_grad_sum <= tolerance_grad:
                return loss

            # variables cached in state (for tracing)
            d = local_state.get('d')  #  Negated flat gradient (to start)
            t = local_state.get('t')
            old_dirs = local_state.get('old_dirs')
            old_stps = local_state.get('old_stps')
            H_diag = local_state.get('H_diag')
            prev_flat_grad = local_state.get('prev_flat_grad')
            prev_loss = local_state.get('prev_loss')

            n_iter = 0
            # optimize for a max of max_iter iterations
            while n_iter < max_iter:
                # keep track of nb of iterations
                n_iter += 1
                local_state['n_iter'] += 1

                ############################################################
                # compute gradient descent direction
                ############################################################
                if local_state['n_iter'] == 1:
                    d = flat_grad.neg()
                    old_dirs = []
                    old_stps = []
                    H_diag = 1
                else:
                    # do Sdlbfgs update (update memory)
                    y = flat_grad.sub(prev_flat_grad)
                    s = d.mul(t)
                    ys = y.dot(s)  # y*s

                    # update scale of initial Hessian approximation
                    gamma = max(y.dot(y) / s.dot(y), delta)
                    H_diag = 1 / gamma
                    sT_H_inv_s = gamma * s.dot(s)

                    if ys < 0.25 * sT_H_inv_s:
                        theta = 0.75 * sT_H_inv_s / (sT_H_inv_s - ys)
                    else:
                        theta = 1
                    y_bar = theta * y + (1 - theta) * gamma * s

                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)

                    # store new direction/step
                    old_dirs.append(s)
                    old_stps.append(y_bar)

                    # compute the approximate (SdL-BFGS) inverse Hessian
                    # multiplied by the gradient
                    num_old = len(old_dirs)

                    if 'ro' not in state:
                        local_state['ro'] = [None] * history_size
                        local_state['al'] = [None] * history_size
                    ro = local_state['ro']
                    al = local_state['al']

                    for i in range(num_old):
                        ro[i] = 1. / old_stps[i].dot(old_dirs[i])

                    # iteration in SdL-BFGS loop collapsed to use just one buffer
                    q = flat_grad.neg()
                    for i in range(num_old - 1, -1, -1):
                        al[i] = old_dirs[i].dot(q) * ro[i]
                        q.add_(-al[i], old_stps[i])

                    # multiply by initial Hessian
                    # r/d is the final direction
                    d = r = torch.mul(q, H_diag)
                    for i in range(num_old):
                        be_i = old_stps[i].dot(r) * ro[i]
                        r.add_(al[i] - be_i, old_dirs[i])

                if prev_flat_grad is None:
                    prev_flat_grad = flat_grad.clone()
                else:
                    prev_flat_grad.copy_(flat_grad)
                prev_loss = loss

                ############################################################
                # compute step size
                ############################################################
                # reset initial guess for step size
                if weight_decay > 0:
                    d = self._add_weight_decay(weight_decay, d)

                d = d / d.norm()

                if lr_decay:
                    t = lr / (local_state['n_iter'] ** 0.5)
                else:
                    if local_state['n_iter'] == 1:
                        t = min(1., 1. / abs_grad_sum) * lr
                    else:
                        t = lr

                # directional derivative
                gtd = flat_grad.dot(d)  # g * d
                updates.append((t, d))


                ############################################################
                # check conditions
                ############################################################
                if n_iter == max_iter:
                    break

                if current_evals >= max_eval:
                    break

                if abs_grad_sum <= tolerance_grad:
                    break

                if gtd > -tolerance_change:
                    break

                if d.mul(t).abs_().sum() <= tolerance_change:
                    break

                if closure is not None:
                    if abs(loss - prev_loss) < tolerance_change:
                        break

            local_state['d'] = d
            local_state['t'] = t
            local_state['old_dirs'] = old_dirs
            local_state['old_stps'] = old_stps
            local_state['H_diag'] = H_diag
            local_state['prev_flat_grad'] = prev_flat_grad
            local_state['prev_loss'] = prev_loss

        self._add_grad(updates)

        return loss
