import numpy as np

from filters.sensors.beacon import Beacon



class BeaconRangeModel:

    def __init__(self, beacons:list[Beacon], std:np.ndarray, idxs:np.ndarray, d:int):
        
        assert len(beacons) == len(std)

        ## KNOWN BEACON POSITIONS
        self.beacons = np.array(beacons)
        self.idxs = idxs

        ## MEASUREMENT NOISE COVARIANCE
        self.n = len(beacons)
        self.m = np.zeros(self.n)
        self.R = std**2 * np.eye(self.n)
        self.H = np.zeros((self.n, d))
    

    def update(self, obs, x, z) -> tuple:
        ## ESTIMATE Z
        z_est = self.f(x[self.idxs], self.beacons)

        ## UPDATE TRANSITION MATRIX
        self.H[:, self.idxs] = self.d(x[self.idxs], self.beacons, z_est)

        ## ESTIMATE y
        y = z - z_est[:,0] + self.H @ x
        return y, self.H, self.R


    def f(self, x: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.linalg.norm(x - b, axis=-1, keepdims=True)

    def d(self, x, b, z) -> np.ndarray:
        return -(b - x) / z



if __name__ == "__main__":

    x = np.array([1,1,0,0])  # state prior

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
        idxs=np.array([0,1]),
        d=len(x),
    )
    y, H, R = measurent.update({}, x, z)
