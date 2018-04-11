"""
Run a car directly along the track using proportional control
"""

import sys
import math
import logging

import gym
import flatlands

LOGGER = logging.getLogger("flatlands_demo")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def sim_demo():
    """
    Runs along the path contained in the mapfile.
    """

    flatlands_env = gym.make('Flatlands-v0')
    # flatlands = FlatlandsEnv()

    theta = 0
    while True:
        for _ in flatlands_env.world.map_data:

            action = {
                "accel": 0.5,
                "wheel_angle": theta,
            }

            obs = flatlands_env.step(action)

            flatlands_env.render()

            # x and y distance to the 3rd point ahead (in meters)
            point = obs["dist_upcoming_points"][3]

            # x and y form a right triangle, the angle towards which we want to go is their atan
            theta = math.atan(point[0] / point[1])

        flatlands_env.reset()


if __name__ == "__main__":
    LOGGER.debug("Registering flatlands env with Gym")

    LOGGER.info("Starting flatlands demo")
    sim_demo()