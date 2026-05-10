import numpy as np



class RangeSensor:

    def __init__(self, name:str, std:float, idxs:np.ndarray):
        ## KNOWN BEACON POSITIONS
        self.name = name
        self.idxs = idxs

        ## SENSOR COVARIANCE
        self.var = std**2


    def step(self, obs: dict, x: np.ndarray):
        assert self.name in obs, f"{self.name} missing from obs."
        pos = obs[self.name]
        z = np.linalg.norm(x[self.idxs] - pos[self.idxs], axis=-1, keepdims=True)
        w = np.random.normal(0, self.var)
        return z + w
    

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)
