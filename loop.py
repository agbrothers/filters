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

    from filters.sensors.beacon import BeaconSensor
    from filters.measurement.range import BeaconRangeModel, Beacon
    from filters.controllers.constant import ConstantController
    from filters.controllers.random import RandomController
    from filters.controllers.hero import HeroController
    from filters.dynamics.linear import LinearDynamics2D
    from filters.dynamics.linear_wind import WindyLinearDynamics2D
    from filters.actions.accel import LinearAccel2D as Actions
    from filters.kalman import KalmanFilter


    ## TIME
    # loop_dt = 2.0
    loop_dt = 1.0
    # loop_dt = 0.2
    # loop_dt = 0.1
    dynamics_dt = 0.1
    time = 10
    
    ## SENSORS
    beacons = [
        Beacon( 1, 0),
        Beacon( 0, 2),
        Beacon(-3, 0),
    ]
    beacon_std = np.array([
        0.1,
        0.02,
        0.5,
    ])
    sensors = [BeaconSensor(b, s) for b,s in zip(beacons, beacon_std)]

    ## CONTROLLER 
    # controller = ConstantController(Actions, (0., 0.))
    controller = RandomController(Actions, mn=-0.1, mx=0.1)
    beacon_controller = HeroController(Actions)

    ## STATE PRIOR
    pos = np.array([0.0, 0.0])
    vel = np.array([0.5, 1.0]) 
    x = np.concatenate([pos, vel])
    
    ## UNCERTAINTY PRIOR
    state_std=np.array([0.005, 0.005, 0.05, 0.05])
    P = np.eye(len(x)) * state_std # Prior uncertainty

    ## FILTER
    measurement_std = np.array([
        0.05,
        0.01,
        0.5,
    ])
    wind_dynamics = WindyLinearDynamics2D(state_std, dynamics_dt, wind_vec=[-0.03, -0.03])
    dynamics = LinearDynamics2D(state_std, dynamics_dt)
    measurement = BeaconRangeModel(beacons, measurement_std)
    filter_ = KalmanFilter(
        x, P,
        measurement=measurement,
        dynamics=deepcopy(dynamics),
    )

    ## ENTITIES
    # entity = Entity("Platform", x, deepcopy(dynamics), controller, sensors, filter_)
    entity = Entity("Platform", x, deepcopy(wind_dynamics), controller, sensors, filter_)
    beacon_1 = Entity("B1", np.array([*beacons[0], 0, 0]), deepcopy(dynamics), beacon_controller)
    beacon_2 = Entity("B2", np.array([*beacons[1], 0, 0]), deepcopy(dynamics), beacon_controller)
    beacon_3 = Entity("B3", np.array([*beacons[2], 0, 0]), deepcopy(dynamics), beacon_controller)
    
    loop(
        entities=[entity, beacon_1, beacon_2, beacon_3],
        dt=loop_dt,
        time=time,
        seed=0
    )
