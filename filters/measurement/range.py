import numpy as np



class RangeModel:

    def __init__(self, targets:list[str], std:np.ndarray, idxs:np.ndarray, d:int):
        
        assert len(targets) == len(std)

        ## KNOWN BEACON UNCERTAINTY
        self.targets = dict(zip(targets, std**2))
        self.idxs = idxs

        ## MEASUREMENT NOISE COVARIANCE
        self.n = len(targets)
        self.R = std**2 * np.eye(self.n)
        self.H = np.zeros((self.n, d))
    

    def update(self, obs, x, z) -> tuple:
        ## COMPUTE DISTANCE BASED ON OBSERVATIONS AND ESTIMATED STATE
        pos = np.array([obs[b][self.idxs] for b in self.targets if b in obs])

        ## UNCERTAINTY MATRIX
        k = len(pos)
        R = self.R
        H = self.H
        if k < self.n:
            std = np.array([self.targets[b] for b in self.targets if b in obs])
            R = std * np.eye(k)
            H = self.H[:k, :k]

        ## COMPUTE LINEARIZED MAP BETWEEN x AND y
        z_est = self.f(x[self.idxs], pos)
        H[:, self.idxs] = self.d(x[self.idxs], pos, z_est)
        y = z - z_est[...,0] + H @ x

        return y, H, R


    def f(self, x: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.linalg.norm(x - b, axis=-1, keepdims=True)
    
    def d(self, x, b, z) -> np.ndarray:
        return -(b - x) / z



if __name__ == "__main__":

    ## SENSORS
    from filters.sensors.range import RangeSensor
    from filters.dynamics.linear import State

    x = np.array([1,1,0,0]) # Prior state
    targets = ["A","B","C"]
    std = np.array([0.1, 0.2, 0.3])
    obs = {
        "A": np.array([0,2,0,0]),
        "B": np.array([2,0,0,0]),
        "C": np.array([-1,-1,0,0]),
    }
    sensors = [RangeSensor(name, std) for name,std in zip(targets, std)]
    z = np.concatenate([s(obs, x) for s in sensors])

    measurent = RangeModel(
        targets=targets,
        std=std,
        state=State,
        d=len(x),
    )
    y, H, R = measurent.update(obs, x, z)
