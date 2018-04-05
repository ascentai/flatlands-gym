# -*- coding: utf-8 -*-
"""
Bicycle vehicle model module. Also known as single-track model. Inherits basic functionalities from the PointModel
This model overwrites the single point model with some extensions. Instead of just going towards a direction, it has a
front and back wheel (separated by 'wheelbase') and turns along an arc.
Its pose ([x, y, theta]) represents its rear axle,

Mass, friction, etc. are not included in this model.

Mostly based on this:
https://nabinsharma.wordpress.com/2014/01/02/kinematics-of-a-robot-bicycle-model/
"""

from abc import ABCMeta, abstractmethod
import math
import random
import logging

import numpy as np
from math import pi
import pygame
from gym.spaces import Box

from envs.flatlands_sim import geoutils

LOGGER = logging.getLogger("vehicle")


class IVehicleModel:
    __metaclass__ = ABCMeta

    def __init__(self, x, y, theta=0.0, vehicle_id="Base model", debug=False):
        """
        Interface for vehicle models. All models should implement a reference point which corresponds to its pose,
        velocity, and acceleration. This can be the center of mass or anything else but it should be the egocentric
        reference point of the vehicle.

        :param x: starting coord in meters
        :param y: starting coord in meters
        :param theta: starting heading angle [optional]
        :param vehicle_id: string identifier of this object [optional]
        :param debug: turns on or off debug messages (verbose mode) [optional]
        """
        self._id = vehicle_id
        self._debug = debug
        # save initial pose so that we can reset later
        self._initial_pose = np.array([x, y, theta % (2 * pi)])
        self._pose = self._initial_pose
        self._velocity = 0.0
        self._acceleration = 0.0

    def __str__(self):
        return 'Agent ID: {0} - pose: {1} - velocity: {2} - acceleration {3}'.format(
            self._id, self.pose, self.velocity, self.acceleration)

    #region Properties

    @property
    def sprite(self):
        raise NotImplementedError("Unimplemented abstract base method")

    @property
    def pose(self):
        raise NotImplementedError("Unimplemented abstract base method")

    @property
    def position(self):
        raise NotImplementedError("Unimplemented abstract base method")

    @property
    def orientation(self):
        raise NotImplementedError("Unimplemented abstract base method")

    @property
    def velocity(self):
        raise NotImplementedError("Unimplemented abstract base method")

    @property
    def acceleration(self):
        raise NotImplementedError("Unimplemented abstract base method")

    #endregion

    #region Public methods

    @abstractmethod
    def move_accel(self, a=None, alpha=None):
        raise NotImplementedError("Unimplemented abstract base method")

    @abstractmethod
    def move_velocity(self, v=None, alpha=None):
        raise NotImplementedError("Unimplemented abstract base method")

    #@abstractmethod
    # TODO we used this earlier but marked as deprecated for now, let's check back when we clean up
    def visualize(self, plt=None):
        raise NotImplementedError("Unimplemented abstract base method")

    def reset(self, randomize=0):
        """
        Resets vehicle to its original (intitialized) location-heading and sets velocity and accel to 0

        :param randomize: if set to True, x-y placement will randomized and won't be exactly the original x-y
        """
        if randomize > 0:
            x, y, theta = self._initial_pose
            # this is in projected meters
            rnd = random.uniform(-randomize, randomize)
            x += rnd
            rnd = random.uniform(-randomize, randomize)
            y += rnd
            # as of now, we don't need theta noise/randomization here, it is handled in the simulator
            self._pose = np.array([x, y, theta])
        else:
            # else, no randomize, just place back to original
            self._pose = self._initial_pose
        self._velocity = 0.0
        self._acceleration = 0.0

    def set(self, x, y, theta, randomize=0):
        """
        Sets vehicle to a location-heading and sets velocity and accel to 0
        Similar to reset as it will zero out accel and velocity but you can provide an arbitrary location

        :param x: X coord in meters
        :param y: Y coord in meters
        :param theta: heading andle in radians
        """
        self._velocity = 0.0
        self._acceleration = 0.0

        rnd = random.uniform(-randomize, randomize)
        x += rnd
        rnd = random.uniform(-randomize, randomize)
        y += rnd
        self._pose = np.array([x, y, theta % (2 * pi)])

    @abstractmethod
    def get_info_object(self):
        raise NotImplementedError("Unimplemented abstract base method")

    #endregion

    #region Protected methods

    @abstractmethod
    def _set_pose(self, x, y, theta):
        raise NotImplementedError("Unimplemented abstract base method")

    #@abstractmethod
    # TODO we used this earlier but marked as deprecated for now, let's check back when we clean up
    def _update_pose(self, x, y, theta):
        raise NotImplementedError("Unimplemented abstract base method")

    #endregion


class PointModel(IVehicleModel):
    def __init__(self, x, y, theta=0.0, max_velocity=0.5, max_accel=0.1, vehicle_id="Point model", noise=0, **kwargs):

        super().__init__(x, y, theta, vehicle_id=vehicle_id)

        # private members
        self._max_velocity = max_velocity
        self._max_accel = max_accel
        self._noise = noise
        self._previous_theta = theta % (pi * 2)
        self._pose = np.array([x, y, theta % (pi * 2)])

        # construct visual representation of model

        self._sprite = (200, pygame.Surface((1000, 1000), pygame.SRCALPHA, 32))
        car_corners = [(500, 240), (620, 760), (380, 760)]
        pygame.draw.aalines(self._sprite[1], (0, 0, 0), True, car_corners)
        pygame.draw.polygon(self._sprite[1], (0, 0, 0), car_corners)

    @property
    def pose(self):
        """
        Center of mass pose: [x, y, theta]

        :return: The raw pose np.array in x-y-theta format (where x-y is projected lat-lon in meters, Japan projection)
        """
        return self._pose

    @property
    def position(self):
        """
        Center of mass position: [x, y]

        :return: The raw position np.array in projected x-y format (meters, Japan projection)
        """
        return self._pose[0:2]

    @property
    def orientation(self):
        """
        Center of mass orientation: theta
        """
        return self._pose[2]

    @property
    def velocity(self):
        """
        Vehicle velocity
        """
        return self._velocity

    @property
    def acceleration(self):
        """
        Vehicle acceleration (speed diference in the previous two steps)
        """
        return self._acceleration

    @property
    def max_accel(self):
        """
        Maximum acceleration.
        """
        return self._max_accel

    @property
    def max_velocity(self):
        """
        Maximum speed.
        """
        return self._max_velocity

    @property
    def angular_velocity(self):
        """
        The angular velocity based on the last movement
        (change of orientation/heading between steps since we operate step-based).

        :returns: a velocity value (angle/step)
        """
        return self.orientation - self._previous_theta

    @property
    def sprite(self):
        """Get the visual representation of the model."""
        return self._sprite

    #region Public methods

    def move_accel(self, a=None, theta=None):
        """
        Acceleration-based step simulation.
        Vehicle will move forward based on the input parameters (and its internal constraints)

        :param  a:      acceleration with which the model should move
        :param  theta:  angle in which direction the model should move

        :return: None
        """
        if a is None:
            a = self.acceleration
            LOGGER.debug("No acceleration provided, keeping previous value")
        elif self.max_accel is not None:
            # else constrain it to be within the specified min-max
            a = np.clip(a, -self.max_accel, self.max_accel)

        if theta is None:
            theta = self.orientation
            LOGGER.debug("No steer angle provided, keeping previous value")

        v = self.velocity + a

        # generate noise
        rand_v = random.uniform(v * (-self._noise / 100), v * (self._noise / 100))
        rand_t = random.uniform(theta * (-self._noise / 10000), theta * (self._noise / 10000))

        LOGGER.debug("noise values:    v: {0}  t: {1}".format(rand_v, rand_t))

        v += rand_v
        theta += rand_t

        # only constrain velocity if there is a max value specified
        if self.max_velocity is not None:
            v = np.clip(v, 0, self.max_velocity)

        # use geoutils to calculate new position
        new_x, new_y = geoutils.offset(self._pose, v, theta)

        # set new pose
        self._set_pose(new_x, new_y, theta)
        # accel = new velo - old velo
        # don't use the user supported one in the params as it might be larger than the limit!
        self._acceleration = v - self._velocity
        # save new velo
        self._velocity = v

    def move_velocity(self, v, theta=None):
        """
        Velocity-based step simulation.
        Vehicle will move forward based on the input parameters (and its internal constraints)

        :param  v:      velocity/speed with which the model should move
        :param  theta:  angle in which direction the model should move

        :return: None
        """
        accel = v - self.velocity
        self.move_accel(accel, theta)

    def get_info_object(self):
        car_info_object = {
            "car_model": "Point",
            "object_type": "car",
            "car_position_x": self.position[0],
            "car_position_y": self.position[1],
            "car_direction": self.orientation,
            "car_speed": self.velocity,
            "car_accel": self.acceleration,
            "max_speed": self.max_velocity,
            "max_accel": self.max_accel,
        }

        return car_info_object

    #endregion

    #region Private methods

    def _set_pose(self, x, y, theta):
        """
        Sets (saves) a new pose.

        :param  x:      new latitude (in meters, projected)
        :param  y:      new longitude (in meters, projected)
        :param  theta:  new heading (angle)

        :return: None
        """
        # save old pose so we can calculate speed and accel
        prev = self._pose

        # constrain theta onto [0..2*pi]
        if theta < 0:
            new_theta = theta % (2 * pi)
            LOGGER.debug("Converting {} degrees to {} degrees.".format(theta, new_theta))
            theta = new_theta
        elif theta > 2 * pi:
            new_theta = theta % (2 * pi)
            LOGGER.debug("Converting {} degrees to {} degrees.".format(theta, new_theta))
            theta = new_theta

        self._previous_theta = self.orientation
        self._pose = np.array([x, y, theta])

        LOGGER.debug("Vehicle pose set " + str(self))

    #endregion


class BicycleModel(PointModel):
    """
    Represents one instance of the bicycle model. Holds its state variables and capable to execute its actions.
    It inherits a lot of functionalities from the simpler PointModel.
    """

    # Toyota Corolla has 2.6m wheelbase
    # 50 m/s max speed = 180 kmph
    # WGS84 is in meters, let's keep use meters for now
    def __init__(
            self,
            x,
            y,
            theta=0.0,
            wheelbase=2.6,
            track=1.2,
            max_wheel_angle=math.pi / 3,  # 60 degrees
            max_velocity=0.5,
            max_accel=0.1,
            vehicle_id="Bicycle model",
            noise=0):

        super().__init__(x, y, theta, vehicle_id=vehicle_id, max_velocity=max_velocity, max_accel=max_accel)

        # private members
        self._wheelbase = wheelbase
        self._track = track
        self._max_wheel_angle = max_wheel_angle % math.pi
        self._wheel_turn_angle = 0.0
        self._noise = noise
        self._previous_wheel_angle = 0.0
        self._num_observed_poins = 5

        LOGGER.info("===== Bicycle vehicle model initialized with the following parameters ====")
        LOGGER.info(self)

    #region Properties
    @property
    def wheelbase(self):
        """Get the length of the vehicle in meters (wheelbase)"""
        return self._wheelbase

    @property
    def track(self):
        """Get the width of the vehicle in meters (track)"""
        return self._track

    @property
    def wheel_turn_angle(self):
        """
        Current wheel turn angle

        :returns: the current wheel turn angle (in degrees, a value between -max_wheel_angle...max_vmax_wheel_angle)
        """
        return self._wheel_turn_angle

    @property
    def max_wheel_angle(self):
        """
        The max wheel angle - since the wheel is symmetric, the min wheel angle is just -max_wheel_angle.

        :returns: the current max wheel angle
        """
        return self._max_wheel_angle

    @property
    def turn_radius(self):
        """
        Current turn radius based on the current wheel turn angle

        :returns: a radius in meters. None if we are not turning, be sure to handle corner case.
        """
        if self.wheel_turn_angle is None or self.wheel_turn_angle == 0.0:
            return None

        return self._wheelbase / math.tan(self.wheel_turn_angle)

    @property
    def wheel_angle_change(self):
        """
        The difference between the current and last recorded wheel angles

        :returns: the cange value in degrees (signed)
        """
        return self._wheel_turn_angle - self._previous_wheel_angle

    @property
    def center_of_turn(self):
        """
        The coordinate of the point we are turning around (i.e. the center of the circle along which we are turning)

        :returns: an x-y coordinate pair. None if we are not turning, be sure to handle corner case.
        """
        if self._wheel_turn_angle == 0.0:
            return None

        x, y, theta = self.pose
        x_center = x + self.turn_radius * math.cos(theta)
        y_center = y - self.turn_radius * math.sin(theta)

        return (x_center, y_center)

    @property
    def radial_speed(self):
        """
        The sideways component of our speed vector

        :returns: a velocity value (meters / step)
        """
        return self.velocity * math.sin(self.orientation)

    @property
    def cross_radial_speed(self):
        """
        The forward component of our speed vector (along the axis of the vehicle)

        :returns: a velocity value (meters / step)
        """
        return self.velocity * math.cos(self.orientation)

    @property
    def sprite(self):
        """Get the visual representation of the model."""
        return self._sprite

    #endregion

    #region IVehicleModel implementation

    def move_accel(self, a=None, wheel_angle=None):
        """
        Acceleration-based step simulation.
        Vehicle will move forward based on the input parameters (and its internal constraints)

        :param  a:            acceleration with which the model should move
        :param  wheel_angle:  angle in which direction the wheel should be turned before movement

        :return: None
        """

        if a is None:
            a = self.acceleration
            LOGGER.debug("No acceleration provided, keeping previous value: %f", self.acceleration)
        elif self.max_accel is not None:
            # else constrain it to be within the specified min-max
            a = np.clip(a, -self.max_accel, self.max_accel)
        if wheel_angle is None:
            wheel_angle = self.wheel_turn_angle
            LOGGER.debug("No steer angle provided, keeping previous value: %f", self.wheel_turn_angle)
        elif self._max_wheel_angle is not None:
            # constrain angle within allowed boundaries
            wheel_angle = np.clip(wheel_angle, -self._max_wheel_angle, self._max_wheel_angle)

        # generate noise on the inputted control parameters
        rand_accel = random.uniform(a * (-self._noise / 100), a * (self._noise / 100))
        rand_wheel_angle = random.uniform(wheel_angle * (-self._noise / 100), wheel_angle * (self._noise / 100))
        LOGGER.debug("Added action noise values: acceleration: {0}  wheel_angle: {1}".format(
            rand_accel, rand_wheel_angle))

        # add generated noise (it'll be 0 if noise is set to 0)
        a += rand_accel
        wheel_angle += rand_wheel_angle

        self._previous_wheel_angle = self._wheel_turn_angle
        self._wheel_turn_angle = wheel_angle
        v = self.velocity + a

        # constrain velocity within allowed boundaries
        if self.max_velocity is not None:
            v = np.clip(v, 0, self.max_velocity)

        # calculate the changes in position and heading
        # v is the distance traveled by the rear wheel during this step
        if self.turn_radius is not None:
            beta = v / self.turn_radius
            theta = self.orientation + beta
        else:
            theta = self.orientation

        if self.center_of_turn is None:
            # just move forward
            # use geoutils to do the straight line movement
            x_prime, y_prime = geoutils.offset(self.position, v, theta)
        else:
            xc, yc = self.center_of_turn
            x_prime = xc - self.turn_radius * math.cos(theta)
            y_prime = yc + self.turn_radius * math.sin(theta)

        # set new pose
        self._set_pose(x_prime, y_prime, theta)
        # accel = new velo - old velo
        # don't use the user supported one in the params as it might be larger than the limit!
        self._acceleration = v - self._velocity
        # save new velo
        self._velocity = v

    def move_velocity(self, v, wheel_angle=None):
        """
        Velocity-based step simulation.
        Vehicle will move forward based on the input parameters (and its internal constraints)
        The implementation is the same as in the parent but reimplemented since the angle represents something else here

        :param  v:            velocity/speed with which the model should move
        :param  wheel_angle:  angle in which direction the wheels should point [-max_angle .. max_angle]

        :return: None
        """
        accel = v - self.velocity
        self.move_accel(accel, wheel_angle)

    def get_info_object(self):
        car_info_object = {
            "car_model": "Bicycle",
            "object_type": "car",
            "car_position_x": self.position[0],
            "car_position_y": self.position[1],
            "car_direction": self.orientation,
            "steering_angle": self.wheel_turn_angle,
            "car_speed": self.velocity,
            "car_accel": self.acceleration,
            "max_wheel_angle": self.max_wheel_angle,
            "max_speed": self.max_velocity,
            "max_accel": self.max_accel,
            "wheelbase": self.wheelbase
        }

        return car_info_object

    def observation_space_continuous(self):
        # get the model constraints
        w = self.max_wheel_angle

        v = self.max_velocity
        a = self.max_accel

        # min and max observation distances of coming up points from the "sensor"
        # multiplied by 2 because x-y
        # TODO create an actual sensor model and move the whole sensing logic there
        min_dists = [-100] * (self._num_observed_poins * 2)
        max_dists = [100] * (self._num_observed_poins * 2)

        # populate pose boundaries based on settings in flatlands_params
        pose_min = [0,0]
        pose_max = [1,1]

        # no need to map to [-1..1] interval
        # first array in box are the min values, second array is the max values
        box = Box(
            np.array([
                *min_dists,
                0,  # velocity
                -a,  # acceleration
                -w,  # wheel angle
                0  # distance from path
            ]),
            np.array([
                *max_dists,
                v,  # velocity
                a,  # acceleration
                w,  # wheel angle
                2000  # distance from path
            ]))

        return Box(box.low, box.high)

    def action_space (self):
        # get the model contstraints
        w = self.max_wheel_angle
        w = 2 * math.pi
        a = self.max_accel

        return Box(np.array([-a, -w]), np.array([a, w]))
    #endregion
