import numpy as np
from enum import IntEnum

from filters.actions.accel import LinearAccel2D as Actions



class State(IntEnum):
    POS_X = 0
    POS_Y = 1
    VEL_X = 2
    VEL_Y = 3


class WindyLinearDynamics2D:

    def __init__(self, std: np.ndarray, dt: float, 
                 wind_vec: list[int,int], **kwargs):
        ## INIT CONTROLS
        n = len(std)
        self.u = np.zeros(n)
        self.wind = np.array([0, 0, *wind_vec])
        self.dt = dt
        
        ## PROCESS NOISE COVARIANCE
        self.Q = std**2 * np.eye(n)

        ## DYNAMICS TRANSITION MATRIX
        self.F = np.array([
           #[p1 p2 v1 v2]
            [ 0, 0, 1, 0],
            [ 0, 0, 0, 1],
            [ 0, 0, 0, 0],
            [ 0, 0, 0, 0],
        ])
        self.PHI = np.eye(n) + self.F * dt 

        ## PLOTTING
        self.x_hist = []


    def reset(self):
        ## RESET HISTORY
        self.x_hist = [self.x_hist[0]]
        self.u = np.zeros(len(self.x))


    def update(self, actions:Actions, dt:float) -> tuple:
        ## STATE-SPACE CONTROLS
        self.u[State.VEL_X.value] = actions.x
        self.u[State.VEL_Y.value] = actions.y
        return self.PHI, self.Q, self.u


    def step(self, actions:Actions, time:float):
        ## VALIDATE TIMESTEP
        assert time >= self.dt and time / self.dt % 1 == 0, \
            "Dynamics `dt` must cleanly divide step `time`"
        
        ## DYANMICS UPDATE(S)
        self.update(actions, self.dt) 
        ## Zero-order hold
        for t in range(int(time / self.dt)):
            x = self.PHI @ self.x + self.u + self.wind
            self.x_hist.append(x)
        return x


    def get_PHI(self):
        assert self.PHI is not None, "Must call update() first"
        return self.PHI
    
    def get_Q(self):
        return self.Q
    
    def get_u(self):
        return self.u

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    @property
    def x(self):
        return self.x_hist[-1]
    


if __name__ == "__main__":

    ## ICs
    pos = np.array([0.0, 0.0])
    vel = np.array([0.5, 1.0]) 
    x = np.concatenate([pos, vel])
    wind_vec = (-0.01, -0.02)

    ## UNCERTAINTY
    std=np.array([0.0, 0.0, 0.1, 0.1])

    ## DYNAMICS
    dt = 0.01
    dynamics = WindyLinearDynamics2D(std, dt, wind_vec)

    ## UPDATE
    x = dynamics.step(x, Actions(0., 0.), time=0.1)

    ## VALIDATE EXPECTED STATE
    np.testing.assert_allclose(x[State.POS_X.value], 0.0455, atol=1e-10)
    np.testing.assert_allclose(x[State.POS_Y.value], 0.091,  atol=1e-10)
    np.testing.assert_allclose(x[State.VEL_X.value], 0.4,    atol=1e-10)
    np.testing.assert_allclose(x[State.VEL_Y.value], 0.8,    atol=1e-10)
