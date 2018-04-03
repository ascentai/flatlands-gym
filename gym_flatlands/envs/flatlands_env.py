"""
Gym environment for a on-track driving simulator
"""

import sys
import logging
import random

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
#from box import Box

from envs.flatlands_sim import DrawMap, BicycleModel, WorldMap

LOGGER = logging.getLogger("flatlands_env")


class FlatlandsEnv(gym.Env):
    """
    Gym environment for on-track driving simulator
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Load the track, draw module, etc.
        """
        self.world = WorldMap(
                "gym_flatlands/envs/flatlands_sim/original_circuit_green.csv")
        self.draw_class = DrawMap(map_data=self.world.map_data)
        self.vehicle_model = BicycleModel(*self.world.path[0],
                self.world.direction[0], max_velocity=1)

        self.car_info = None

        if (not hasattr(self.vehicle_model, "max_wheel_angle") or not
                hasattr(self.vehicle_model, "wheel_turn_angle")):
            print("*** Error: vehicle model must simulate steerable wheels.")
            sys.exit()

        self.observation_space = spaces.Box(
            np.array([
                *([-100] * 10),                         # distances
                0,                                      # velocity
                -self.vehicle_model.max_accel,          # acceleration
                -self.vehicle_model.max_wheel_angle,    # wheel angle
                0                                       # distance to path
            ]),
            np.array([
                *([100] * 10),                      # distances
                self.vehicle_model.max_velocity,    # velocity
                self.vehicle_model.max_accel,       # acceleration
                self.vehicle_model.max_wheel_angle, # wheel angle
                2000                                # distance to path
            ])
        )

        self.action_space = spaces.Box(
            np.array([
                -self.vehicle_model.max_accel,      # acceleration
                -self.vehicle_model.max_wheel_angle # wheel angle
            ]),
            np.array([
                self.vehicle_model.max_accel,       # acceleration
                self.vehicle_model.max_wheel_angle  # wheel angle
            ])
        )

    def step(self, action):
        """
        Accepts an `action` object, consisting of desired accelleration (accel)
        and the steering angle

        Returns on observation object
        """

        accel = action[0]
        wheel_angle = action[1]

        self.vehicle_model.move_accel(accel, wheel_angle)

        dists = self.world.get_dist_upcoming_points(
                self.vehicle_model.position, self.vehicle_model.orientation)

        accumulator = []
        for t in dists:
            accumulator.append(t[0])
            accumulator.append(t[1])

        d_path = self.world.distance_from_track(
                self.vehicle_model.position)

        goal_dist = self.world.distance_to_goal(self.vehicle_model.position)
        total_dist = self.world.path_length

        reward = d_path * -10 + total_dist / goal_dist
        obs = [
            *accumulator,                           # distances
            self.vehicle_model.velocity,            # velocity
            self.vehicle_model.acceleration,        # acceleration
            self.vehicle_model.wheel_turn_angle,    # wheel angle
            d_path                                  # distance to path
        ]

        return obs, reward, 0, 0

    def reset(self):
        """
        Reset the car to a static place somewhere on the track.
        """

        LOGGER.debug("system resetting")

        idx = random.randint(0, len(self.world.path) - 1)
        LOGGER.debug("Randomly placing the vehicle near map point #{}".format(idx))
        x, y = self.world.path[idx]
        theta = self.world.direction[idx]
        self.vehicle_model.set(x, y, theta)

        self.distance_traveled = 0
        
        return self.step([0, 0])[0]

    def render(self, mode='human', close=False):
        """
        Use pygame to draw the map
        """

        car_info_object = self.vehicle_model.get_info_object()
        self.draw_class.draw_car(car_info_object)
