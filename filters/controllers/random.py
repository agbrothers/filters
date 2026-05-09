import numpy as np


class RandomController:
    
    def __init__(self, action_cls, mn=-1, mx=1, **kwargs):
        self.action_cls = action_cls
        self.actions_len = len(action_cls._fields)
        self.mn = mn
        self.mx = mx
        self.rng = mx - mn

    def step(self, obs, x, dt):
        actions = np.random.random(self.actions_len) * self.rng + self.mn
        return self.action_cls(*actions)

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)
