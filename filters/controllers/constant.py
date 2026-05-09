import numpy as np


class ConstantController:
    
    def __init__(self, action_cls, actions, **kwargs):
        self.action_cls = action_cls
        self.actions = actions

    def step(self, obs, x, dt):
        return self.action_cls(*self.actions)

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)
