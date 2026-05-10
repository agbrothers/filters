import numpy as np
from tqdm import trange
from copy import deepcopy

from filters.entity import Entity
from filters.render import render_history


def loop(entities:list[Entity], dt:float, time:int, seed:int):

    np.random.seed(seed)

    ## LOOP THROUGH TIME
    for t in trange(int(time/dt)):

        ## COMPUTE OBSERVATION OF ENTITY STATES
        obs = {e.name: e.x for e in entities}

        ## STEP ENTITIES
        for entity in entities:
            entity.step(obs, dt)

    ## GATHER TRAJECTORIES
    true_position_history = {e.name: np.array(e.dynamics.x_hist)[:,:2] for e in entities}
    filter_position_history = {e.name: np.array(e.filter.x_hist)[:-1,:2] for e in entities if e.filter is not None}
    filter_uncertainty_history = {e.name: np.array(e.filter.P_hist)[:-1,:2,:2] for e in entities if e.filter is not None}
    filter_residual_history = {e.name: np.array(e.filter.residual_hist) for e in entities if e.filter is not None}
    filter_residual_std_history = {e.name: np.array(e.filter.S_hist) for e in entities if e.filter is not None}
    
    ## SAVE PLAYBACK VIDEO
    render_history(
        true_position_history,
        filter_position_history,
        filter_uncertainty_history,
        filter_residual_history,
        filter_residual_std_history,
        output_path="videos/simulation.mp4",
        fps=20,
        padding=3,
        expand_alpha=0.2,
        residual_cols=3,
        traj_fig_height=8.0,
        dpi=150,
        # dpi=500,
    )
    return



if __name__ == "__main__":

    from filters.sensors.range import RangeSensor
    from filters.sensors.beacon import BeaconSensor
    from filters.measurement.range import RangeModel
    from filters.measurement.beacon import BeaconRangeModel, Beacon
    from filters.controllers.constant import ConstantController
    from filters.controllers.random import RandomController
    from filters.controllers.orbit import OrbitController
    from filters.controllers.hero import HeroController
    from filters.dynamics.linear import LinearDynamics2D, State
    from filters.dynamics.linear_wind import WindyLinearDynamics2D
    from filters.actions.accel import LinearAccel2D as Actions
    from filters.kalman import KalmanFilter


    ## TIME
    # sensor_dt = 2.0
    sensor_dt = 1.0
    # sensor_dt = 0.2
    # sensor_dt = 0.1
    dynamics_dt = 0.1
    time = 30
    
    ## CONTROLLER 
    beacon_controller = HeroController(Actions)
    controller = RandomController(Actions, mn=-0.1, mx=0.1)
    # controller = OrbitController(Actions, omega=0.1)
    # controller = ConstantController(Actions, (0., 0.))

    ## STATE PRIOR
    pos = np.array([0.5, 0.5])
    vel = np.array([0.5, 1.0]) 
    x = np.concatenate([pos, vel])
    x_prior = x + np.array([1,1,0,0]) ## Biased prior

    ## UNCERTAINTY PRIOR
    dynamics_std=np.array([0.01, 0.01, 0.1, 0.1])
    prior_std=np.array([2.00, 2.00, 2.0, 2.0])
    P = np.eye(len(x)) * prior_std # Prior uncertainty

    ## BUILD DYNAMICS MODEL
    # wind_dynamics = WindyLinearDynamics2D(dynamics_std, dynamics_dt, wind_vec=[-0.03, -0.03])
    dynamics = LinearDynamics2D(dynamics_std, dynamics_dt)

    ## BUILD BEACON ENTITIES
    beacon_1 = Entity("B1", np.array([ 1, 0, 0, 0]), deepcopy(dynamics), beacon_controller)
    beacon_2 = Entity("B2", np.array([ 0, 2, 0, 0]), deepcopy(dynamics), beacon_controller)
    beacon_3 = Entity("B3", np.array([-3, 0, 0, 0]), deepcopy(dynamics), beacon_controller)
    beacons = [beacon_1.name, beacon_2.name, beacon_3.name]
    
    ## BUILD MEASUREMENT MODEL
    idxs = np.array([State.POS_X.value, State.POS_Y.value])
    beacon_std      = np.array([0.1, 0.02, 0.5]) ## REAL SENSOR NOISE
    # beacon_std      = np.array([0.01, 0.01, 0.01]) ## REAL SENSOR NOISE
    # measurement_std = np.array([0.95, 0.95, 0.95]) ## FILTER SENSOR NOISE ASSUMPTION
    # measurement_std = np.array([0.05, 0.05, 0.05]) ## FILTER SENSOR NOISE ASSUMPTION
    measurement_std = np.array([0.1, 0.01, 0.5]) ## FILTER SENSOR NOISE ASSUMPTION
    measurement = RangeModel(beacons, measurement_std, idxs, len(x))
    # measurement = BeaconRangeModel(beacons, measurement_std, idxs, len(x))

    ## BUILD SENSORS
    sensors = [RangeSensor(name, std, idxs) for name,std in zip(beacons, beacon_std)]
    # sensors = [BeaconSensor(b, s) for b,s in zip(beacons, beacon_std)]

    ## BUILD FILTER
    filter_ = KalmanFilter(
        x_prior, P,
        measurement=measurement,
        dynamics=deepcopy(dynamics),
    )

    ## ENTITIES
    entity = Entity("Platform", x, deepcopy(dynamics), controller, sensors, filter_)
    # entity = Entity("Platform", x, deepcopy(wind_dynamics), controller, sensors, filter_)

    loop(
        entities=[entity, beacon_1, beacon_2, beacon_3],
        dt=sensor_dt,
        time=time,
        seed=0
    )
