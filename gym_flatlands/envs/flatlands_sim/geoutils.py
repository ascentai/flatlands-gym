"""
Geometric calculation utility functions
"""

from math import sin, cos, atan2, pi
from collections import namedtuple

from shapely import geometry as geom
from pyproj import Proj, transform


def distance(prev, curr):
    """
    Calculates the distance between two points, assuming a equal x-y projection

    :param  prev:   point "a"
    :param  curr:   point "b"

    :return: the distance in meters (or whatever the input units were)
    """

    point1 = geom.Point(prev)
    point2 = geom.Point(curr)

    return point1.distance(point2)


def bearing(prev, curr):
    """
    Calculates the heading necessary to get from prev to curr.

    :param  prev:   2-tuple of x,y coordinates
    :param  curr:   2-tuple of x,y coordinates

    :return: the required heading to reach curr
    """

    # This gives us the angle in radians , from the positive x-axis
    angle_rad = atan2(curr[1] - prev[1], curr[0] - prev[0])

    # we're taking bearing from the positive y-axis, so rescale it
    angle = pi / 2 - angle_rad

    return angle


def offset(point, dist, angle):
    """
    Calculate new coordinates based on a starting coordinate, distance in meters
    with an absolute heading angle going from positive-y axis.

    :param  point:  a 2-tuple of x-y coordinates
    :param  dist:   the distance to "travel"
    :param  angle:  the heading from positive-y

    :return: a 2=tuple whose first position is the new x, and whose second position is the new y
    """

    new_x = point[0] + dist * sin(angle)
    new_y = point[1] + dist * cos(angle)

    return (new_x, new_y)


def relative_distance(point1, point2, angle):
    """
        Find the relative distance between two points, given the origin and the angle it's facing.

        Accepts:
            origin, point: Two x-y tuples of the input and destination
            angle: int/float of the angle the origin is facing (y-extrusion direction)
        Returns:
            A 2-tuple containing x,y distance in meters from the origin, to the destination using
            the axis created by extruding the y-axis along the input angle
        """

    #angle between current point, and next point
    direction_angle = bearing(point1, point2)

    # angle going from car direction to the direction angle
    direct_angle = direction_angle - angle

    # shortest difference between car direction angle and desired angle to the next point
    heading_angle = min(direct_angle, 2 * pi - direct_angle)

    # Get information between our points
    point1 = geom.Point(point1)
    point2 = geom.Point(point2)
    absolute_dist = point1.distance(point2)

    # Then use trig to find the relative x and y distances between the points
    x_dist = absolute_dist * sin(heading_angle)
    y_dist = absolute_dist * cos(abs(heading_angle))

    # Create our return value as a namedtuple (accessible by index or dot-notation)
    distance_tuple = namedtuple("distance_tuple", ["distances", "heading"])
    relative = distance_tuple((x_dist, y_dist), heading_angle)
    return relative


def proj_to_local(points, new_proj="epsg:30176"):
    """
    Convert from global geographic coordinates to a reference x-y coordinate set
    centered in Japan (epsg:4326 -> epsg:30176)

    For other available projections see https://epsg.io/

    Accepts:
        points: A list of 2-tuples containing lat-long data (y-x format)
        new_proj: A string denoting a new projection to cast points to
            formatted like `epsg:{proj_number}`
    Returns:
        A numpy array of the same size as the input list with 2-tuples containing
            relative (x,y) projcetion coordinates
    """

    global_projection = Proj(init="epsg:4326")
    jp_projection = Proj(init=new_proj)
    local_coord = namedtuple('local_coord', 'x_local, y_local')

    local_paths = []
    for point in points:
        projection_x, projection_y = transform(global_projection, jp_projection, point[1], point[0])
        local_paths.append(local_coord(projection_x, projection_y))

    return local_paths