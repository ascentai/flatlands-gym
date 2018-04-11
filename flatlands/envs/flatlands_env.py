"""
Gym environment for a on-track driving simulator
"""

import sys
import logging
import random
from os import path

import gym

from .flatlands_sim import DrawMap, BicycleModel, WorldMap

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

        map_file = path.join(sys.prefix, "flatlands/original_circuit_green.csv")

        self.world = WorldMap(map_file)
        self.draw_class = DrawMap(world=self.world)
        self.vehicle_model = BicycleModel(*self.world.path[0], self.world.direction[0], max_velocity=1)

        self.car_info = None

    def step(self, action):
        """
        Accepts an `action` object, consisting of desired accelleration (accel)
        and the steering angle

        Returns on observation object
        """

        accel = action["accel"]
        wheel_angle = action["wheel_angle"]

        self.vehicle_model.move_accel(accel, wheel_angle)

        obs = {
            "reward":
            0,
            "dist_upcoming_points":
            self.world.get_dist_upcoming_points(self.vehicle_model.position, self.vehicle_model.orientation),
        }

        return obs

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

    def render(self, mode='human', close=False):
        """
        Use pygame to draw the map
        """

        car_info_object = self.vehicle_model.get_info_object()
        self.draw_class.draw_car(car_info_object)
