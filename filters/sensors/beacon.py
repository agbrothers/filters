import numpy as np

from typing import NamedTuple



class Beacon(NamedTuple):
    px: float
    py: float


class BeaconSensor:

    def __init__(self, beacon:Beacon, std:float):
        ## KNOWN BEACON POSITIONS
        self.pos = np.array(beacon)

        ## SENSOR COVARIANCE
        self.var = std**2


    def step(self, obs: np.ndarray, x: np.ndarray):
        z = np.linalg.norm(x[:2] - self.pos, axis=-1, keepdims=True)
        w = np.random.normal(0, self.var)
        return z + w
    

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)
