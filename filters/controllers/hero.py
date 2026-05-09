class HeroController:
    
    def __init__(self, action_cls, **kwargs):
        self.action_cls = action_cls
        self.actions = [0] * len(action_cls._fields)

    def step(self, obs, x, dt):
        return self.action_cls(*self.actions)

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)
