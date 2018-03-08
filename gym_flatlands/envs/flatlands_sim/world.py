"""
Module for handling world information and generating world related observations
Usage as follows:
    from map import Map
"""

import csv
from collections import namedtuple
import math
import logging

import shapely.geometry as geom
from scipy.spatial import cKDTree as KDTree

from envs.flatlands_sim import geoutils

MAP_POINT = namedtuple('map_point', [
    'lat',
    'lon',
    'width',
    'direction',
    'segment_length',
])

LAT_LONG = namedtuple('lat_long', 'lat, long')
CAR_POSITION_POINT = namedtuple("car_position_ref", ["lat_long", "direction"])
LOCAL_COORD = namedtuple("local_coord", "x_local, y_local")

LOGGER = logging.getLogger("world")


class WorldMap(object):
    """
    Generic implimentation of a representation of a map in python

    Impliments method for loading, visualizing, etc. map files which
    contain GPS coords, and other misc map-related data like track width.

    The custom load() function must be implimented specifically for your data format,
    which should populate the `self.map_data` variable with a list of namedtuples
    formatted with the map_point structure
    """

    def __init__(self, track_file=None, zoomed_percentage_of_window=0.3, debug=False, *args, **kwargs):

        # list of mapping points to be filled after loading data
        self.map_data = None
        # will be calculated when the map is loaded in load()
        self._path_length = None

        # Projected path data to be filled after loading data
        self.projected_path = None

        # This will determine if we should display stuff like text on the screen
        self.debug = debug
        self.font = None

        # Location of the input file used by load function
        self.map_file = track_file

        self.lat_long = LAT_LONG

        # Store the current position of the car
        self.car_position_ref = CAR_POSITION_POINT
        self.car_position = None

        self._model = None

        # Holds the scipy kd-tree created from all map-points in a x-y projection
        self.kd_tree = None

        # Draw class (can't initialize until we have loaded our data)
        self.zoomed_percentage_of_window = zoomed_percentage_of_window

        # Load the map data
        self.load(track_file)
        self.post_load(project_to_local=False)

        LOGGER.debug("Map initialized.")

    @property
    def model(self):
        """Gets the vehicle model."""
        return self._model

    @model.setter
    def model(self, value):
        """Sets the vehicle model."""
        self._model = value

    @property
    def path_global(self):
        """
        Returns all of the lat-long coords in the map file
        """
        return [(x.lat, x.lon) for x in self.map_data]

    @property
    def path(self):
        """
        Returns x, y coordinates from a local projection
        """
        return [(x.x_local, x.y_local) for x in self.projected_path]

    @property
    def path_length(self):
        """
        Returns the total length of the path
        """
        return self._path_length

    @property
    def width(self):
        """
        Returns a list of all the widths for every point in the track
        """

        return [(x.width) for x in self.map_data]

    @property
    def start(self):
        """
        Returns only the first set of lat-long coords in the map file
        """
        return (self.projected_path[0].x_local, self.projected_path[0].y_local)

    @property
    def goal(self):
        """
        Returns only the last set of lat-long coords for the map file
        """
        return (self.projected_path[-1].x_local, self.projected_path[-1].y_local)

    @property
    def segment_length(self):
        """
        Returns the distances of each segment towards the next one
        """
        return [(x.segment_length) for x in self.map_data]

    @property
    def direction(self):
        """
        Return the direction of a segment of track
        """
        return [(x.direction) for x in self.map_data]

    @property
    def lat_min(self):
        """
        Find the minimum latitude of the track
        """
        return min([x.y_local for x in self.projected_path])

    @property
    def lat_max(self):
        """
        Find the maximum latitude of the track
        """
        return max([x.y_local for x in self.projected_path])

    @property
    def long_min(self):
        """
        Find the minimum longitude of the track
        """
        return min([x.x_local for x in self.projected_path])

    @property
    def long_max(self):
        """
        Find the maximum longitude of the track
        """
        return max([x.x_local for x in self.projected_path])

    def load(self, track_file):
        """
        Loads a given custom track
        """
        self.map_file_path = track_file

        try:
            LOGGER.debug("Attempting to open %s", self.map_file_path)
            with open(self.map_file_path, newline="") as csvfile:
                spamreader = csv.reader(csvfile, delimiter=",", quotechar='"')
                map_data = []
                line = csvfile.readline()
                scale = float(line[line.index("=") + 1:])

                line = csvfile.readline()
                height = float(line[line.index("=") + 1:])

                line = csvfile.readline()

                for row in spamreader:
                    row_float = [float(i) for i in row]

                    map_data.append(
                        MAP_POINT(
                            lon=row_float[0] / scale,
                            # flip the y-coordinate in preparation for the rotation in draw.py
                            lat=(height - row_float[1]) / scale,
                            width=row_float[2],
                            direction=row_float[4],
                            segment_length=row_float[3] / scale))

                end = map_data[-1]
                start = map_data[0]

                theta = geoutils.bearing((end.lon, end.lat), (start.lon, start.lat))
                dist = math.sqrt(abs(end.lon - start.lon)**2 + abs(end.lat - start.lat)**2)
                map_data[-1] = MAP_POINT(
                    lon=end.lon, lat=end.lat, width=end.width, direction=theta, segment_length=dist)
                map_data.append(start)
                LOGGER.debug("Found %d points of track data", len(map_data))
                self.map_data = map_data
        except EnvironmentError:
            LOGGER.error("Failed to import file %s", self.map_file_path)
            LOGGER.error(EnvironmentError)
            raise

    def post_load(self, project_to_local=True):
        """
        After loading data, call this function to initialize the rest of the
        map class for things such as the local coordinate system
        """

        if project_to_local:
            # Converts our path data to EPSG 30176 x-y space
            # List of namedtuples with x_local, and y_local attributes
            LOGGER.debug("Generating projection of path")
            self.projected_path = geoutils.proj_to_local(self.path_global)
        else:
            local_coord = namedtuple("local_coord", "x_local, y_local")
            self.projected_path = [local_coord(i[1], i[0]) for i in self.path_global]

        # Scipy kd_tree for efficient lookup of points (like nearest neighbor)
        LOGGER.debug("Generating KD-tree of projection")
        self.kd_tree = KDTree(self.projected_path)

        # precalculate path length here to save time later
        dists = (i for i in self.segment_length)
        self._path_length = sum(dists)

    def distance_from_track(self, input_location):
        """
        Returns the distance from the track for a set of geographic coordinates

        Accepts: input_location: 2-tuple containing latitute and longitude

        Returns: the distance in meters from the track
        """

        LOGGER.debug("Searching for the nearest point on the track to %s", input_location)

        # Find the closest points
        nearest_index = self.get_nearest_points(input_location, one_point_only=True, return_index=True)
        LOGGER.debug("The nearest point is at index %s", nearest_index)

        # Go forward and back from this point to get two more
        point1 = self.kd_tree.data[nearest_index - 1]
        if nearest_index - 1 > len(self.kd_tree.data):
            point2 = self.kd_tree.data[nearest_index]
            point3 = self.kd_tree.data[nearest_index + 1]
        else:
            point2 = self.kd_tree.data[0]
            point3 = self.kd_tree.data[1]

        # Draw lines between them, and find the closest point on the lines to the input
        line1 = geom.LineString([point1, point2])
        line2 = geom.LineString([point2, point3])

        # Convert the input point to our local projection system
        point = geom.Point(input_location)

        # Now find the closest point on each line to our input
        point_on_line1 = line1.interpolate(line1.project(point))
        point_on_line2 = line2.interpolate(line2.project(point))

        # Now get the distances between those points and the input point
        dist1 = point.distance(point_on_line1)
        dist2 = point.distance(point_on_line2)

        LOGGER.debug("The distance to the nearest two line segments on the track is %s, %s", dist1, dist2)

        # We're only concerned about the smaller one, so we'll return it
        return min(dist1, dist2)

    def distance_to_goal(self, input_location):
        """
        Computes the distance around the track to the goal point

        Accepts:
            input_location: a tuple containing lat-long coordinates formatted to epsg:30176
        Returns:
            A float containing the distance in meters to the goal point
        """

        # First get the closest point
        closest_point_idx = self.get_nearest_points(input_location, one_point_only=True, return_index=True)

        dists = (i for i in self.segment_length[closest_point_idx:])

        return sum(dists) + self.distance_from_track(input_location)

    def get_dist_upcoming_points(self, position, angle, num_points=5):
        """
        Function for finding the relative location of the upcoming points on
        the track nearest to the target.

        Accepts:
            position: lat-long tuple containing the search point
            n: the number of points to return distance info for
        Returns:
            A list of (n) tuples containing the distance in meters (x and y)
            to each of the upcoming points on the track. Positive numbers are right and front.

        Function works by searching for the nearest point on the track, and then calculating the
        relative distance to this and the next n points in the path. If the first point has a
        negative y-direction then this point is dropped, since it is probably
        behind the car
        """

        # Get the closest point to the input
        nearest_point_idx = self.get_nearest_points(position, one_point_only=True, return_index=True)
        LOGGER.debug("Nearest point to the input is %s", nearest_point_idx)
        LOGGER.debug("input:%s, closest:%s", position, self.kd_tree.data[nearest_point_idx - 1])

        # Get the upcoming points on the track
        last_point = nearest_point_idx + num_points + 1
        point_set = self.projected_path[nearest_point_idx:last_point]

        # For proper "looping" around the track, go back to the beggining if we run out of values
        for i in range(num_points - len(point_set) + 1):
            point_set.append(self.path[i])

        # Get the distance to each of these points
        distances = [geoutils.relative_distance(position, destination, angle) for destination in point_set]
        LOGGER.debug("Full distances set: %s", distances)

        # If the first value is behind the origin then don't return it
        if distances[0].distances[1] < 0:
            return [x.distances for x in distances[1:]]
        return [x.distances for x in distances[:-1]]

    def get_nearest_points(self, origin, one_point_only=False, return_index=False):
        """
        Find the nearest two points on the track to an arbitrary lat-long pair,

        Accepts:
            origin: a lat-long tuple containing coordinates formatted to epsg:30176
            one_point_only: A boolean to ask for only a single value instead of two
            return_index: A boolean for getting the nearest point in Map.map_data
            instead of its coords

        Returns:
            Two 2-tuples (or 1 with one_point_only), containing the lat-long for the
            nearest point(s) on the track
        """

        closest = self.kd_tree.query(origin)
        idx = closest[1]
        closest_local_coords = self.kd_tree.data[idx - 1]

        if one_point_only is False:

            # Catch situation where the closest point is at the end of the track
            if idx == len(self.path) - 1:
                LOGGER.debug("Closest point is the last in the track, taking first instead")

                if return_index:
                    return idx, 0

                return closest_local_coords, self.path[0]

            # Index isn't max, so return closest point, plus the next one
            if return_index:
                return idx, idx + 1

            return closest_local_coords, self.path[idx + 1]

        if return_index:
            return idx
        return self.path[idx]
