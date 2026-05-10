import numpy as np

from filters.actions.accel import LinearAccel2D as Actions


class KalmanFilter:

    def __init__(self, x, P, measurement, dynamics):
        ## MODELS
        self.measurement = measurement
        self.dynamics = dynamics
        self.dt = dynamics.dt

        ## HISTORY
        self.P_hist = [P]
        self.x_hist = [x]
        self.y_hist = []
        self.S_hist = []
        self.residual_hist = []


    def reset(self):
        ## RESET HISTORY
        self.P_hist = [self.P_hist[0]]
        self.x_hist = [self.x_hist[0]]
        self.y_hist = []
        self.S_hist = []
        self.residual_hist = []


    def step(self, actions:Actions, time:float, obs=None, z=None):
        ## VALIDATE TIMESTEP
        assert time >= self.dt and time / self.dt % 1 == 0, \
            "Filter `dt` must cleanly divide step `time`"

        ## DYANMICS UPDATE(S)
        PHI, Q, u = self.dynamics.update(actions, self.dt)
        for t in range(int(time / self.dt)):
            x_new, P_new = self.dynamics_update(PHI, Q, u)
        
        ## MEASUREMENT UPDATE
        if obs is not None and z is not None:
            x_new, P_new = self.measurement_update(obs, z, x_new, P_new)
        return x_new, P_new


    def dynamics_update(self, PHI, Q, u):
        ## PROPAGATE DYNAMICS
        x_new = PHI @ self.x + u
        P_new = PHI @ self.P @ PHI.T + Q
        self.x_hist.append(x_new)
        self.P_hist.append(P_new)
        return x_new, P_new


    def measurement_update(self, obs, z, x=None, P=None):
        if x is None: x = self.x
        if P is None: P = self.P

        ## INFERENCE MEASUREMENT MODEL
        y, H, R = self.measurement.update(obs, x, z)

        ## COMPUTE KALMAN GAIN (JOSEPH FORM)
        S = H @ P @ H.T + R
        K = (P @ H.T) @ np.linalg.inv(S)

        ## MEASUREMENT UPDATE
        residual = y - H @ x
        x_new = x + K @ residual
        P_new = P - K @ H @ P
        # IKH = np.eye(len(x)) - K @ H
        # P_new = IKH @ P @ IKH.T + K @ R @ K.T
        # P_new = 0.5 * (P + P.T)

        ## UPDATE HISTORY
        self.x_hist[-1] = x_new
        self.P_hist[-1] = P_new
        self.S_hist.append(np.sqrt(np.diag(S)))
        self.residual_hist.append(residual)
        return x_new, P_new
    
    @property
    def P(self):
        return self.P_hist[-1]
    
    @property
    def x(self):
        return self.x_hist[-1]

    @property
    def y(self):
        return self.y_hist[-1]

    @property
    def residual(self):
        return self.residual_hist[-1]

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)
    


if __name__ == "__main__":

    from filters.sensors.beacon import BeaconSensor
    from filters.dynamics.linear import LinearDynamics2D, State
    from filters.measurement.beacon import BeaconRangeModel, Beacon

    ## STATE PRIOR
    pos = np.array([0.0, 0.0])
    vel = np.array([0.5, 1.0]) 
    x = np.concatenate([pos, vel])
    idxs = np.array([State.POS_X.value, State.POS_Y.value])
    
    ## UNCERTAINTY PRIOR
    state_std=np.array([0.01, 0.01, 0.1, 0.1])
    P = np.eye(len(x)) * state_std # Prior uncertainty

    beacons = [
        Beacon( 1, -0.1),
        Beacon( 0, 2),
        Beacon(-3, 0.1),
    ]
    beacon_std = np.array([
        0.01,
        0.02,
        0.03,
    ])
    

    ## SENSOR MEASUREMENT
    obs = {}
    sensors = [BeaconSensor(b, s) for b,s in zip(beacons, beacon_std)]
    z = np.concatenate([s(obs, x) for s in sensors])

    ## BUILD FILTER
    measurement = BeaconRangeModel(beacons, beacon_std, idxs, len(x))
    dynamics = LinearDynamics2D(state_std, dt=0.1)
    kf = KalmanFilter(
        x, P,
        measurement=measurement,
        dynamics=dynamics,
    )

    ## STEP FILTER
    kf.step(Actions(0.0, 0.0), time=1, z=z)



    ## RANGE VERSION
    from filters.sensors.range import RangeSensor
    from filters.measurement.range import RangeModel

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

    ## BUILD FILTER
    measurent = RangeModel(targets, std, idxs, len(x))
    kf = KalmanFilter(
        x, P,
        measurement=measurement,
        dynamics=dynamics,
    )
    ## STEP FILTER
    kf.step(Actions(0.0, 0.0), time=1, z=z)

