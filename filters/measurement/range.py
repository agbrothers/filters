import numpy as np

from filters.sensors.beacon import Beacon



class BeaconRangeModel:

    def __init__(self, beacons:list[Beacon], std:np.ndarray):
        
        assert len(beacons) == len(std)

        ## KNOWN BEACON POSITIONS
        self.beacons = np.array(beacons)

        ## MEASUREMENT NOISE COVARIANCE
        k = len(beacons)
        self.m = np.zeros(len(std))
        self.R = std**2 * np.eye(k)
        self.H = None
    

    def update(self, x, z) -> tuple:
        ## STATE-SPACE TRANSITION MATRIX
        z_est = self.f(x, self.beacons)

        ## COMPUTE LINEARIZED BETWEEN x AND y
        H = self.d(x, self.beacons, z_est)
        if H.shape[-1] < len(x):
            mismatch = len(x) - H.shape[1]
            pad = np.zeros((H.shape[0], mismatch))
            H = np.concat((H, pad), axis=1)

        y = z - z_est[:,0] + H @ x
        self.H = H
        return y, self.H, self.R

    def f(self, x: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.linalg.norm(x[:2]-b, axis=-1, keepdims=True)
    
    def d(self, x, b, z) -> np.ndarray:
        return -(x[:2] - b) / z

    def get_H(self) -> np.ndarray:
        assert self.H is not None, "Must call update() first"
        return self.H
    
    def get_R(self) -> np.ndarray:
        return self.R
    
    def get_w(self) -> np.ndarray:
        return np.random.multivariate_normal(self.m, self.R)



if __name__ == "__main__":

    x = np.array([1,1,0,0]) # Prior state

    ## BEACONS
    beacons = [
        Beacon( 1, 0),
        Beacon( 0, 2),
        Beacon(-3, 0),
    ]
    beacon_std = np.array([
        0.1,
        0.2,
        0.3,
    ])

    ## SENSORS
    from filters.sensors.beacon import BeaconSensor
    sensors = [BeaconSensor(b, s) for b,s in zip(beacons, beacon_std)]
    z = np.concatenate([s(None, x) for s in sensors])

    measurent = BeaconRangeModel(
        beacons,
        beacon_std,
    )
    y, H, R = measurent.update(x, z)
