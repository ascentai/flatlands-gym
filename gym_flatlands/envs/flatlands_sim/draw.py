"""
This module is used for visualizing the map and vehicle, using pygame
"""

from collections import namedtuple
import math
from math import pi
import logging

import pygame
from pygame import gfxdraw

from envs.flatlands_sim.world import WorldMap
from envs.flatlands_sim import geoutils

LOGGER = logging.getLogger("draw")


class DrawMap():
    """
    A class for visualizing system state with pygame.
    """

    def __init__(self, map_data, map_path):

        custom_map = WorldMap(map_path)

        self.map_data = map_data
        self.projected_path = custom_map.projected_path

        self.path = [(x[0], x[1]) for x in self.projected_path]
        self.segment_length = [(x.segment_length) for x in self.map_data]

        self.y_min = custom_map.y_min
        self.y_max = custom_map.y_max
        self.x_min = custom_map.x_min
        self.x_max = custom_map.x_max

        # The width isn't set, it's generated based on the aspect ratio of the map data
        pygame.init()
        max_height = pygame.display.Info().current_h - 50
        self.window_y = min(1024, max_height)
        self.window_x = None

        # self.ascent_logo = pygame.image.load("map/ascent_logo.png")
        # self.ascent_logo_reduced = pygame.transform.scale(self.ascent_logo, (25, 25))

        pygame.font.init()
        self.font = pygame.font.Font(None, 40)

        self.car_position = None
        self.car_direction = None

        # self.debug = debug

        # Screen holds the pygame window to be written on by components
        self.screen = None

        # Define a named tuple to use for displaying our map data
        self.map_point_screen = namedtuple('map_point_screen', [
            'screen_x_y',
            'segment_length',
        ])

        # Define our (soft) border we have inside the window
        self.border_width = 0.05
        self.border_size = self.window_y * self.border_width

        #For the zoomed view in the corner
        self.zoomed_window_size = custom_map.zoomed_percentage_of_window * self.window_y

        # Hold the track information so we don't have to re-draw it every refresh
        self.track_draw_info = None

        # Distance (in m) to look around each side of the car, so box edge is x2 minimap distance
        self.minimap_distance = 15

        # Car information
        self.steering_angle = None
        self._car_sprite = None
        self._wheelbase = None

    def _pre_draw(self):
        """
        Determining how to draw and scale everything on the track is an intensive operation,
        so let's do it once, and then save all the information that we need to reproduce it
        when redrawing the screen
        """

        self.track_draw_info = {}
        LOGGER.debug("track bounding box: (%s, %s) (%s, %s)", self.x_min, self.y_min, self.x_max, self.y_max)

        # top_left = geom.Point(self.x_min, self.y_max)
        # top_right = geom.Point(self.x_max, self.y_max)
        # bottom_left = geom.Point(self.x_min, self.y_min)

        screen_height_in_m = geoutils.distance((self.x_min, self.y_max), (self.x_min, self.y_min))
        screen_width_in_m = geoutils.distance((self.x_min, self.y_max), (self.x_max, self.y_max))

        LOGGER.debug("screen height: %sm, width %sm", screen_height_in_m, screen_width_in_m)

        # Scale the width of the window to match the aspect ratio of the track
        self.window_x = int(self.window_y * (screen_width_in_m / screen_height_in_m))

        # Spawn our pygame window
        # pygame.display.set_icon(self.ascent_logo)
        self.screen = pygame.display.set_mode((self.window_x, self.window_y))
        pygame.display.set_caption("Flatlands Sim")

        # Get our list of corner_sets to draw each segment of the track
        self.track_draw_info["scaled_corners"] = self._scale_corners()

        # The info about our centerlines and section names
        scaled_points = self._scale_for_display(self.path)
        LOGGER.debug("orig start: %s", self.path[0])
        LOGGER.debug("scaled start: %s", scaled_points[0])

        self._scale_sprite(2.6)

        self.track_draw_info["points_scaled"] = [
            self.map_point_screen(x[0], x[1]) for x in zip(scaled_points, self.segment_length)
        ]

    def _scale_sprite(self, car_wheelbase):
        """
        For any given car, create a scaled sprite for display.

        Applies the new surface to self._car_sprite

        Accepts: Nothing
        Returns: Nothing
        """

        # This block taken from the vehicle model and moved to it's own function
        self._car_sprite = (200, pygame.Surface((1500, 1500), pygame.SRCALPHA, 32))

        # We're only using the wheelbase (length) currently, so let's keep
        # the x/y scale of the car constant, rather then scale seperately

        reference_wheelbase = 2.6
        car_scale_factor = car_wheelbase / reference_wheelbase

        standard_x_dist = 240
        standard_y_dist = 540

        scaled_x_dist = car_scale_factor * standard_x_dist
        scaled_y_dist = car_scale_factor * standard_y_dist

        mid_pt = 750

        x_lower = mid_pt - scaled_x_dist / 2
        x_upper = mid_pt + scaled_x_dist / 2

        y_lower = mid_pt - scaled_y_dist / 2
        y_upper = mid_pt + scaled_y_dist / 2

        car_corners = [(x_lower, y_lower), (x_upper, y_lower), (x_upper, y_upper), (x_lower, y_upper)]

        black = (0, 0, 0)
        pygame.gfxdraw.aapolygon(self._car_sprite[1], car_corners, black)
        pygame.gfxdraw.filled_polygon(self._car_sprite[1], car_corners, black)

        # Bottom left, top, bottom right
        tri_corners = [(mid_pt - 125, y_lower - 50), (mid_pt, y_lower - 250), (mid_pt + 125, y_lower - 50)]

        red = (255, 0, 0)
        pygame.gfxdraw.aapolygon(self._car_sprite[1], tri_corners, red)
        pygame.gfxdraw.filled_polygon(self._car_sprite[1], tri_corners, red)

    def draw(self, update_screen=True):
        """
        Uses standardized self.map_data from load function and displays
        it on the screen. Be sure to run self.load() in your init function
        before calling this draw function, or their will be no data to work with
        """

        # Scale all of our data before displaying it, if it hasn't been done already
        if self.track_draw_info is None:
            self._pre_draw()

        # Draw our background, and all our track segments
        self.screen.fill((247, 247, 247))
        for corner in self.track_draw_info["scaled_corners"]:
            gfxdraw.aapolygon(self.screen, corner, (204, 204, 204))
            gfxdraw.filled_polygon(self.screen, corner, (204, 204, 204))

        last_point = self.track_draw_info["points_scaled"][0]

        # for idx, point in enumerate(self.track_draw_info["points_scaled"][1:]):

        #     # if self.debug:
        #     #     # Draw each point as a dot
        #     #     gfxdraw.circle(self.screen, *point.screen_x_y, 1, (0, 0, 0))
        #     #     gfxdraw.filled_circle(self.screen, *point.screen_x_y, 1, (0, 0, 0))

        #     #     # label each section on the track where a reset occurs (no segment lenth)
        #     #     if last_point.segment_length == 0:
        #     #         text = self.font.render("Section %s" % idx, True, (0, 128, 0)) #yapf:disable
        #     #         self.screen.blit(text, last_point.screen_x_y)

        #     # elif last_point.segment_length != 0:
        #     #     # Draw the centerline
        #     #     pygame.draw.aaline(self.screen, (238, 102, 102), last_point.screen_x_y, point.screen_x_y)

        #     last_point = point

        self.draw_window_trimmings()

        if update_screen:
            pygame.display.flip()

    def draw_window_trimmings(self):
        """
        Window trimming is text/ graphics to display on the window, unrelated to contents.

        This function will generate the trimmings, and then blit the screen to render them.abs

        Accepts: Nothing
        Returns: Nothing
        """

        red = (255, 0, 0)
        top_left = (0, 5)
        text = self.font.render("Flatlands Simulator", True, red)

        self.screen.blit(text, top_left)

    def _draw_car_info(self, car_info):
        """
        This function will graph the car info onto a new pygame surface, and then return it.

        Accepts:
            car_info: A list containing tuples of the info to be visualized, where each tuple contains:
                a label (0), a value (1), a 2-tuple containing lower and upper bounds (2)

        Returns:
            A surface that can be drawn with pygame
        """

        # Draw the background
        info_view = pygame.Surface((  #pylint: disable=E1121
            self.zoomed_window_size, self.zoomed_window_size))

        # This will make the background transparent
        info_view.set_colorkey((255, 255, 255))
        info_view.fill((255, 255, 255))

        # Draw the charts
        info_view = self._render_info_view_charts(car_info=car_info, surface=info_view)

        return info_view

    def _render_info_view_charts(self, car_info, surface):
        """
        Draws charts related to car info onto a surface provided.

        Accepts:
            car_info: A list of tuples, see format in function `self._draw_car_info`
            surface: the surface to modify

        Returns:
            A pygame surface, a modified version of the input one
        """

        # Display related constants
        edge_gap = 0.1
        bar_height = 0.1
        bar_seperation_factor = 1

        bar_border_px = self.zoomed_window_size * edge_gap
        bar_height_px = self.zoomed_window_size * bar_height

        font = pygame.font.Font(None, 20)
        red = (255, 0, 0)

        for idx, (label, value, (min_range, max_range)) in enumerate(car_info):

            # This is the factor to seperate each bar chart by (top->top distance)
            y_adjustment = (1 + bar_seperation_factor) * idx * bar_height_px

            # Write the label for the chart (above the bar chart)
            text = font.render(label, True, red)
            text_location = (bar_border_px, bar_height_px + y_adjustment - 13)
            surface.blit(text, text_location)

            # Draw the borders of the chart
            top_left = (bar_border_px, bar_height_px + y_adjustment)
            bottom_left = (bar_border_px, bar_height_px * 2 + y_adjustment)
            bottom_right = (self.zoomed_window_size - bar_border_px, bar_height_px * 2 + y_adjustment)
            top_right = (self.zoomed_window_size - bar_border_px, bar_height_px + y_adjustment)

            chart_corners = ([top_left, top_right, bottom_right, bottom_left])
            pygame.draw.aalines(surface, (0, 0, 0), True, chart_corners)

            # Write the text for the limits of the charts
            low_label = font.render(str(min_range)[:4], True, red)
            low_label_location = (bar_border_px - 25, bar_height_px * 1.3 + y_adjustment)
            surface.blit(low_label, low_label_location)

            upper_label = font.render(str(max_range)[:4], True, red)
            upper_label_location = (self.zoomed_window_size - bar_border_px + 2, bar_height_px * 1.3 + y_adjustment)

            surface.blit(upper_label, upper_label_location)

            # Plot the input value
            value_pct = (value + abs(min_range)) / (max_range - min_range)

            x_loc = bar_border_px + value_pct * (self.zoomed_window_size - 2 * bar_border_px)

            top = (x_loc, bar_height_px + y_adjustment)
            bottom = (x_loc, bar_height_px * 2 + y_adjustment)

            pygame.draw.line(surface, red, top, bottom, 2)

            # Plot the Text of the value (below the bar chart)
            value_text = font.render(str(value)[:4], True, red)
            value_text_location = (x_loc + 5, bar_height_px * 2 + y_adjustment)
            surface.blit(value_text, value_text_location)

        return surface

    def _scale_corners(self):
        """
        Returns the four corners of range of segments of track
        """

        if "raw_corners" in self.track_draw_info:
            corners = self.track_draw_info["raw_corners"]

        else:
            corners = []
            for idx in range(0, len(self.map_data)):
                corners.append(self._get_corners(idx))
            self.track_draw_info["raw_corners"] = corners

        scaled_corners = [self._scale_for_display(corner) for corner in corners]

        return scaled_corners

    def shutdown(self):
        """shutdown visuals"""

        pygame.display.quit()
        pygame.quit()

    def draw_car(self, kwargs):
        """
        Draws the car at a set (new) position on the map

        Accepts:    kwargs: A dict containing values needed for drawing the car
                        ex. car_position_x, car_directions, etc.

        Return: Nothing
        """

        # check for quit event in pygame events queueu
        if any([event.type == pygame.QUIT for event in pygame.event.get()]):
            LOGGER.info("Pygame window closed, stopping simulation")
            self.shutdown()

        if self.track_draw_info is None:
            self.draw()

        self.car_position = (kwargs["car_position_x"], kwargs["car_position_y"])
        self.car_direction = kwargs["car_direction"]
        try:
            self.steering_angle = kwargs["steering_angle"]
        except KeyError:
            pass

        LOGGER.debug("Front of car located at %s", self.car_position)

        if self.car_position[0] == float("inf") or self.car_position[1] == float("inf"):
            LOGGER.error("Car is at impossible coordinates, cannot draw")
            return

        if self._wheelbase is None:
            self._wheelbase = kwargs["wheelbase"]
            self._scale_sprite(self._wheelbase)

        # refresh the screen to overwrite the previous frame
        self.draw(update_screen=False)

        # requisite proportions for proper scaling of the sprite
        mPerPx = (self.y_max - self.y_min) / (self.window_y - self.border_size * 2)

        pxPerM, car_sprite = self._car_sprite

        # apparently conversion of the alpha channel is only allowed with an instantiated pygame.display, so we convert
        # it here instead of in the vehicle model
        car_sprite = car_sprite.convert_alpha()

        # perform the rotation first, so we only have to do it once
        rotate_sprite = self._rotate_car(car_sprite, self.car_direction)
        # scale for the large map
        rs_car_sprite = pygame.transform.scale(
            rotate_sprite,
            (int(rotate_sprite.get_width() / pxPerM / mPerPx), int(rotate_sprite.get_height() / pxPerM / mPerPx)))

        # paint the sprite at the model's location
        sprite_rect = rs_car_sprite.get_rect()
        sprite_rect.center = self._scale_for_display([self.car_position])[0]

        # copy the sprite to the screen
        self.screen.blit(rs_car_sprite, sprite_rect)

        # Draw the zoomed in view of the car in the corner
        miniMPerPx = (self.minimap_distance * 2) / self.zoomed_window_size
        mini_rs_car_sprite = pygame.transform.scale(rotate_sprite,
                                                    (int(rotate_sprite.get_width() / pxPerM / miniMPerPx),
                                                     int(rotate_sprite.get_height() / pxPerM / miniMPerPx)))

        zoom_view = self._draw_zoom_view(mini_rs_car_sprite)

        if zoom_view is not None:
            self.screen.blit(zoom_view, (0, self.window_y - self.zoomed_window_size))

        # Info view
        if self.steering_angle == None:
            car_info_to_visualize = [
                ("car_speed", kwargs["car_speed"], (0, kwargs["max_speed"])),
                ("steering_angle", 0, (-1, 1)),
                ("car_accel", kwargs["car_accel"], (-kwargs["max_accel"], kwargs["max_accel"])),
            ]
        else:
            car_info_to_visualize = [
                ("car_speed", kwargs["car_speed"], (0, kwargs["max_speed"])),
                ("steering_angle", self.steering_angle, (-kwargs["max_wheel_angle"], kwargs["max_wheel_angle"])),
                ("car_accel", kwargs["car_accel"], (-kwargs["max_accel"], kwargs["max_accel"])),
            ]

        info_view = self._draw_car_info(car_info_to_visualize)
        self.screen.blit(info_view, (self.window_x - self.zoomed_window_size, 0))

        pygame.display.flip()

    def _rotate_car(self, sprite, angle):
        """
        Performs a rotation transform on the given image.

        :param  sprite: the image to transform
        :param  angle:  how much to rotate, in radians

        :return the rotated sprite
        """
        rotate_rect = sprite.get_rect().copy()
        rotate_sprite = pygame.transform.rotate(sprite, angle * -180 / math.pi)
        rotate_rect.center = rotate_sprite.get_rect().center
        rotate_sprite = rotate_sprite.subsurface(rotate_rect).copy()

        return rotate_sprite

    def _get_corners(self, point_index, only_start_corners=False):
        """
        Accepts information surround an arc, and plots the endpoints
        """

        start_point = self.projected_path[point_index]
        direction = self.map_data[point_index].direction
        width = self.map_data[point_index].width

        corner_1 = geoutils.offset(start_point, width / 2, direction - pi / 2)

        corner_2 = geoutils.offset(start_point, width / 2, direction + pi / 2)

        if only_start_corners:
            return [corner_1, corner_2]

        # If there is a next point to connect to, set it as the termination corners
        elif len(self.map_data) - 1 > point_index \
            and self.map_data[point_index +1].segment_length != 0:

            # next_point = self.map_data[point_index + 1]
            next_corners = self._get_corners(point_index + 1, only_start_corners=True)
            corner_3, corner_4 = next_corners[0], next_corners[1]

        # otherwise extrapolates the corners based on direction and length of the segment
        else:
            LOGGER.debug("No next point found for idx %s, extrapolating instead of connecting", point_index)
            segment_length = self.map_data[point_index].segment_length
            end_point = geoutils.offset(start_point, segment_length, direction)

            corner_3 = geoutils.offset(end_point, width / 2, direction - pi / 2)

            corner_4 = geoutils.offset(end_point, width / 2, direction + pi / 2)

        coords = [x for x in [corner_1, corner_2, corner_4, corner_3]]

        return coords

    def _scale_for_display(self, input_coordinates):
        """
        Scales a set of x-y coordinates to integer values (for location on the display)

            input_coordinates:
        Accepts: input_coordinates, a list of 2-tuples each containing x-y values

        Returns: a new list of the same size as the input list containing integer values for display
        """

        # seperate x and y
        x_points_scaled = scale_list(
            input_list=[x[0] for x in input_coordinates],
            min_val=self.x_min,
            max_val=self.x_max,
            new_range=(self.window_x - self.border_size * 2),
            offset=self.border_size)

        y_points_scaled = scale_list(
            input_list=[x[1] for x in input_coordinates],
            min_val=self.y_min,
            max_val=self.y_max,
            new_range=(self.window_y - self.border_size * 2),
            offset=self.border_size,
            adjust_amount=self.window_y)

        scaled_list = [(x[0], x[1]) for x in zip(x_points_scaled, y_points_scaled)]

        return scaled_list

    def _scale_for_mini_display(self, input_coordinates, idxs=None):
        """
        Similar to the _scale_for_display function except that you can also pass
        in a list of indexes (for the indexes of the input coordinates) so that returned
        values can be looked up in the self.path list later
        """

        return_idxs = True
        if idxs is None:
            return_idxs = False
            idxs = [0 for i in input_coordinates]

        path_and_idxs = list(zip(input_coordinates, idxs))

        top = geoutils.offset(self.car_position, self.minimap_distance, 0)
        right = geoutils.offset(self.car_position, self.minimap_distance, pi / 2)
        bottom = geoutils.offset(self.car_position, self.minimap_distance, pi)
        left = geoutils.offset(self.car_position, self.minimap_distance, 1.5 * pi)

        # seperate x and y
        x_points_scaled = scale_list(
            input_list=[x[0][0] for x in path_and_idxs],
            min_val=left[0],
            max_val=right[0],
            new_range=self.zoomed_window_size,
            offset=0)

        y_points_scaled = scale_list(
            input_list=[x[0][1] for x in path_and_idxs],
            min_val=bottom[1],
            max_val=top[1],
            new_range=self.zoomed_window_size,
            offset=0,
            adjust_amount=self.zoomed_window_size)

        # If either of our scaling funcs have nothing to display, return nothing
        if not all([x_points_scaled, y_points_scaled]):
            return [], [] if return_idxs else []

        scaled_list = [(x[0], x[1]) for x in zip(x_points_scaled, y_points_scaled)]

        idxs_reduced = [x[1] for x in path_and_idxs]

        if return_idxs is False:
            return scaled_list

        return scaled_list, idxs_reduced

    def _draw_zoom_view(self, sprite):
        """
        For drawing the map outline view in the corner

        Accepts:
            sprite : a zoomed view surface
        Returns:
            A pygame `screen` Object that can be drawn at any point on the screen
                OR
            a None object, if the coordinates given are impossible to visualize
                (x-y OOB)
        """

        # Get outerbounds of the zoomed view
        bounds = [
            geoutils.offset(self.car_position, self.minimap_distance, angle) for angle in [0, pi / 2, pi, 1.5 * pi]
        ]
        LOGGER.debug("Bounds of the zoomed view are %s", bounds)

        # Draw box for the zoomed view (coords relative only to the zoom view, NOT absolute to screen)
        window_size_reduced = self.zoomed_window_size - 3
        mini_view_corners_reduced = [(1, 1),
                                     (1, window_size_reduced),
                                     (window_size_reduced,
                                      window_size_reduced),
                                     (window_size_reduced, 1)] #yapf: disable

        zoom_view = pygame.Surface((  #pylint: disable=E1121
            self.zoomed_window_size, self.zoomed_window_size))
        zoom_view.fill((247, 247, 247))

        corners = self.track_draw_info["raw_corners"]
        scaled_corners = list(map(self._scale_for_mini_display, corners))

        LOGGER.debug("Found %d segments to draw in the zoomed view", len(scaled_corners))
        if scaled_corners:
            LOGGER.debug("First corner segment: %s", scaled_corners[0])

            for corner in scaled_corners:
                gfxdraw.aapolygon(zoom_view, corner, (204, 204, 204))
                gfxdraw.filled_polygon(zoom_view, corner, (204, 204, 204))

        # Draw the track midline
        mini_view_screen = self._scale_for_mini_display(self.path)
        LOGGER.debug("Reduced map to %d points for zoomed view", len(mini_view_screen))
        segments = [x for x in self.segment_length]
        mini_track_draw_info = [self.map_point_screen(x[0], x[1]) for x in zip(mini_view_screen, segments)]

        if mini_track_draw_info:
            last_point = mini_track_draw_info[0]

            LOGGER.debug("The first point on the zoomed view (screen) is %s", last_point)
            for point in mini_track_draw_info[1:]:
                if last_point.segment_length != 0:
                    pygame.draw.aaline(zoom_view, (238, 102, 102), last_point.screen_x_y, point.screen_x_y)
                last_point = point

        # paint the sprite in the center of the zoomed view
        car_rect = sprite.get_rect()
        car_rect.center = zoom_view.get_rect().center

        # Draw a box around the outside of the view
        pygame.draw.aalines(zoom_view, (0, 0, 0), True, mini_view_corners_reduced)

        # copy the model to the zoomed view surface (and do it last to make sure it's on top)
        zoom_view.blit(sprite, car_rect)

        return zoom_view


def scale_list(
        input_list,  # pylint: disable=R0913
        min_val,
        max_val,
        new_range: int,
        offset=0,
        adjust_amount=None):
    """
    Scales a list to fit within a new range

    the new range is a number defining the new max, and the offset defines the new minumum

    passing an adjust amount will take every value in the scaled list, and do:
    adjusted_val = (adjust_amount - scaled_amount)

    Returns: a new list, of the same size as the input list containing integer values
    """

    old_range = (max_val - min_val)

    # Check that we won't divide by zero
    if old_range + offset <= 0:
        return

    scaled_list = [int(((old_value - min_val) * new_range) / old_range + offset) for old_value in input_list]

    # For pygames inverted vertical grid system, we'll apply an adjustment if needed
    if adjust_amount is not None:
        scaled_list = [adjust_amount - x for x in scaled_list]

    return scaled_list
