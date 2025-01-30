import math
import numpy as np
import EyelinerWingGeneration
from A import StartMode, SegmentType
from ParentSegment import Segment, point_in_array
from EyelinerWingGeneration import bezier_curve
from StarGeneration import bezier_curve_t


class RandomShapeLineSegment:
    def __init__(self, curviness, curve_direction, curve_location):
        self.curviness = curviness
        self.curve_direction = curve_direction
        self.curve_location = curve_location
        self.points_array = []

    def render(self, start_point, end_point, colour, thickness, ax_n=None):
        num_steps = 50  # Number of points to create a smooth thickness transition/ curve

        if self.curviness>0:
            t_values = np.linspace(0, 1, num_steps)
            P0 = start_point
            P2 = end_point
            direction = EyelinerWingGeneration.normalised_vector_direction(P0, P2)
            P1 = P0 + (self.curve_location * direction) #moves curve_location away from P0 towards P2 relative to length of curve segment
            direction_angle = np.degrees(np.arctan2(direction[1], direction[0]))
            relative_curve_direction_degrees = direction_angle + self.curve_direction
            curve_dir_radians = np.radians(relative_curve_direction_degrees)
            # Calculate x and y offsets
            dx = self.curviness * np.cos(curve_dir_radians)
            dy = self.curviness * np.sin(curve_dir_radians)
            P1 = P1 + np.array([dx, dy])
            self.points_array = np.array([bezier_curve_t(t, P0, P1, P2) for t in t_values])
            x_values, y_values = self.points_array[:, 0], self.points_array[:, 1]
        else:
            x_values = np.linspace(start_point[0], end_point[0], num_steps)
            y_values = np.linspace(start_point[1], end_point[1], num_steps)
            self.points_array = np.column_stack((x_values, y_values))

        if ax_n:
            for i in range(len(x_values) - 1):
                ax_n.plot(
                    [x_values[i], x_values[i + 1]], [y_values[i], y_values[i + 1]],
                    color=colour,
                    linewidth=thickness,
                    solid_capstyle='butt',
                )
        return self.points_array

class RandomShapeSegment(Segment):
    """Line segment with additional properties specific to a line."""
    def __init__(self, segment_type, start, start_mode, end_thickness, relative_angle, colour, bounding_size, corners, lines_list):
        super().__init__(segment_type, start, start_mode, end_thickness, relative_angle, colour)
        self.segment_type = SegmentType.RANDOM_SHAPE
        self.start = start
        self.segment_type = SegmentType.RANDOM_SHAPE
        self.bounding_size = bounding_size
        self.corners = np.array([(corner[0] * self.bounding_size[0], corner[1] * self.bounding_size[1]) for corner in corners])
        #self.lines_segments = [(RandomShapeLineSegment(curviness, curve_direction, curve_location))for line in lines_list]
        self.lines_list = lines_list
        self.points_array = np.array([])  # Initialize as an empty array


    def rotated_corners(self):
        """Rotates edges around the start by """
        angle_rad = math.radians(self.absolute_angle)
        rotated_corners = []
        ox, oy = self.start  # Origin coordinates (start of shape)
        for corner in self.corners:
            x, y = corner
            # Shift point back to the origin (subtract origin from edge)
            x -= ox
            y -= oy
            # Apply the rotation matrix
            new_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
            new_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)

            # Shift the point back to its original position (add origin back)
            new_x += ox
            new_y += oy

            rotated_corners.append((new_x, new_y))

        return rotated_corners

    def move_edges(self):
        movement = EyelinerWingGeneration.un_normalised_vector_direction(self.corners[0], self.start)

        for edge in self.corners:
            edge[0] += movement[0]
            edge[1] += movement[1]

    def render(self, prev_array, prev_angle, prev_colour, prev_end_thickness, ax_n=None):
        if self.start_mode == StartMode.CONNECT and len(prev_array) > 15:
            self.start = (prev_array[-1][0], prev_array[-1][1])
        elif self.start_mode == StartMode.CONNECT and len(prev_array) <= 15:
            end_index = point_in_array(prev_array, 0.5)
            self.start = (prev_array[end_index][0], prev_array[end_index][1])

        if self.start_mode == StartMode.JUMP:
            self.absolute_angle = self.relative_angle
        else:
            self.absolute_angle = (prev_angle + self.relative_angle) % 360

        self.rotated_corners()
        self.move_edges()

        start_point = np.array(self.corners[0])
        #ax_n.scatter(start_point[0], start_point[1], color='red',linewidth=8, zorder=5)
        for i, edge in enumerate(self.corners):
            if i+1 < len(self.corners):
                end_point = np.array(self.corners[i + 1])
                #ax_n.scatter(end_point[0], end_point[1], color='orange', linewidth=3,  zorder=6)
            else:
                end_point = np.array(self.corners[0])
                #ax_n.scatter(end_point[0], end_point[1], color='blue',linewidth=3,  zorder=6)

            section_points_array = self.lines_list[i].render(start_point, end_point, self.colour, self.end_thickness, ax_n)
            self.points_array = np.append(self.points_array,section_points_array)
            start_point = end_point

        #STILL NEED TO PICK ORDR OF EDGES SO THEY JOIN NICELY