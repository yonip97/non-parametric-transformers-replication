from collections import defaultdict
from itertools import chain

import numpy as np
import torch
import warnings
from torch.optim.optimizer import Optimizer
from torch.nn.utils import clip_grad_norm_
from lamb import Lamb


class Lookahead(Optimizer):
    def __init__(self, model_parameters, init_lr, max_steps, end_lr = 0,betas = (0.9,0.999),eps = 1e-6, power = 1,
                 warmup_proportion = None, k=6, alpha=0.5):
        self.optimizer = Lamb(model_parameters,init_lr,betas,eps)
        self.power = power
        self.k = k
        self.alpha = alpha
        self.end_lr = end_lr
        self.init_lr = init_lr
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0
        self.steps_taken = 0
        self.max_steps = max_steps
        self.warmup_steps = warmup_proportion * max_steps
        self.anneal_steps = self.max_steps - self.warmup_steps

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        self.steps_taken += 1
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)


    def flat_then_anneal(self):
        if self.warmup_steps <= self.steps_taken  and self.steps_taken <=self.max_steps:
            steps_post_flat = self.steps_taken - self.warmup_steps
            for group in self.param_groups:
                group['lr'] = 0.5 * self.init_lr*(1+np.cos(np.pi*steps_post_flat/self.anneal_steps))
        elif self.steps_taken > self.max_steps:
            for group in self.param_groups:
                group['lr'] = 0

    def polynomial_decay(self):
        if self.steps_taken < self.warmup_steps:
            lr = float(self.steps_taken) / float(max(1, self.warmup_steps))
        elif self.steps_taken > self.max_steps:
            lr = self.end_lr / self.init_lr  # as LambdaLR multiplies by lr_init
        else:
            lr_range = self.init_lr - self.end_lr
            decay_steps = self.max_steps - self.warmup_steps
            pct_remaining = 1 - (self.steps_taken - self.warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** self.power + self.end_lr
            lr = decay / self.lr_init
        for group in self.param_groups:
            group['lr'] = lr