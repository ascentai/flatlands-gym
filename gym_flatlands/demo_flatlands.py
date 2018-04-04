"""
Run a car directly along the track using proportional control
"""

import sys
import math
import logging

from envs import FlatlandsEnv

LOGGER = logging.getLogger("flatlands_demo")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def sim_demo():
    """
    Runs along the path contained in the mapfile.
    """

    flatlands = FlatlandsEnv()

    theta = 0
    while True:
        for _ in flatlands.world.map_data:

            action = {
                "accel": 0.5,
                "wheel_angle": theta,
            }

            obs = flatlands.step(action)

            flatlands.render()

            # x and y distance to the 3rd point ahead (in meters)
            point = obs["dist_upcoming_points"][3]

            # x and y form a right triangle, the angle towards which we want to go is their atan
            theta = math.atan(point[0] / point[1])

        flatlands.reset()


if __name__ == "__main__":

    LOGGER.info("Starting flatlands demo")
    sim_demo()