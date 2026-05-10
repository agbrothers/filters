import numpy as np


class Entity:

    def __init__(self, name, x, dynamics, controller, sensors=None, filter=None):
        ## COMPONENTS
        self.name = name
        self.filter = filter
        self.sensors = sensors
        self.dynamics = dynamics
        self.controller = controller

        ## PLOTTING
        self.dynamics.x_hist.append(x)


    def reset(self):
        ## RESET HISTORY
        self.dynamics.reset()


    def step(self, obs, dt) -> np.ndarray:
        ## OBSERVE -> MEASUREMENT UPDATE -> CONTROLS -> DYNAMICS UPDATES 
        x = self.x

        if self.filter is not None:
            ## OBSERVE
            z = np.concatenate([s(obs, x) for s in self.sensors])
            ## UPDATE MEASUREMENT MODEL
            x = self.filter.measurement_update(obs, z)

        ## UPDATE CONTROLS 
        actions = self.controller(obs, x, dt)

        ## UPDATE STATE 
        x_new = self.dynamics.step(actions, dt)

        ## UPDATE FILTER
        if self.filter is not None:
            self.filter(actions, dt)

        return x_new


    @property
    def x(self):
        return self.dynamics.x
    


if __name__ == "__main__":
    from filters.sensors.beacon import BeaconSensor
    from filters.actions.accel import LinearAccel2D as Actions
    from filters.measurement.beacon import BeaconRangeModel, Beacon
    from filters.controllers.constant import ConstantController
    from filters.dynamics.linear import LinearDynamics2D, State
    from filters.kalman import KalmanFilter

    ## TIME
    loop_dt = 0.1
    dynamics_dt = 0.01
    time = 10
    
    ## SENSORS
    beacons = [
        Beacon( 1, 0),
        Beacon( 0, 2),
        Beacon(-3, 0),
    ]
    beacon_std = np.array([
        0.01,
        0.02,
        0.03,
    ])
    sensors = [BeaconSensor(b, s) for b,s in zip(beacons, beacon_std)]

    ## CONTROLLER 
    controller = ConstantController(Actions, (0., 0.))

    ## STATE PRIOR
    pos = np.array([0.0, 0.0])
    vel = np.array([0.5, 1.0]) 
    x = np.concatenate([pos, vel])
    idxs = np.array([State.POS_X.value, State.POS_Y.value])
    
    ## UNCERTAINTY PRIOR
    state_std=np.array([0.1, 0.1, 0.1, 0.1])
    P = np.eye(len(x)) * state_std # Prior uncertainty

    ## BUILD FILTER
    dynamics = LinearDynamics2D(state_std, dynamics_dt)
    measurement = BeaconRangeModel(beacons, beacon_std, idxs, len(x))
    filter_ = KalmanFilter(
        x, P,
        dynamics=dynamics,
        measurement=measurement,
    )

    ## BUILD ENTITY
    entity = Entity("Entity", x, dynamics, controller, sensors, filter_)
    
    ## STEP ENTITY
    entity.step(obs={entity.name: x}, dt=loop_dt)




    ## RANGE VERSION
    from filters.sensors.range import RangeSensor
    from filters.measurement.range import RangeModel

    ## CREATE DATA
    targets = ["A","B","C"]
    std = np.array([0.1, 0.2, 0.3])
    obs = {
        entity.name: x,
        "A": np.array([0,2,0,0]),
        "B": np.array([2,0,0,0]),
        "C": np.array([-1,-1,0,0]),
    }
    ## BUILD AND INFERENCE SENSORS
    sensors = [RangeSensor(name, std) for name,std in zip(targets, std)]
    z = np.concatenate([s(obs, x) for s in sensors])

    ## BUILD FILTER
    measurent = RangeModel(targets, std, idxs, len(x))
    kf = KalmanFilter(
        x, P,
        measurement=measurement,
        dynamics=dynamics,
    )

    ## BUILD ENTITY
    entity = Entity("Entity", x, dynamics, controller, sensors, filter_)
    
    ## STEP ENTITY
    entity.step(obs=obs, dt=loop_dt)
