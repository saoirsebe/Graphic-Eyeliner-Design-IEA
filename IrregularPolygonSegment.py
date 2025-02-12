import math
import numpy as np
from matplotlib import pyplot as plt

from A import StartMode, SegmentType, un_normalised_vector_direction, normalised_vector_direction, start_x_range, \
    start_y_range, curviness_range, relative_location_range, random_shape_size_range, corner_initialisation_range, \
    line_num_points
from ParentSegment import Segment, point_in_array
from StarGeneration import bezier_curve_t

def are_points_collinear(points, tolerance=0.5):
    """
    Checks if a set of points are collinear within tolerance.

    points: Nx2 numpy array of (x, y) coordinates.
    tolerance: Maximum allowed deviation from collinearity.
    return: True if points are collinear, False otherwise.
    """
    if len(points) < 3:
        return True  # Two points are always collinear

    # Compute vector differences
    vectors = np.diff(points, axis=0)

    # Compute cross product magnitude (for 2D)
    cross_products = np.cross(vectors[:-1], vectors[1:])

    # If all cross products are close to zero, the points are collinear
    return np.all(np.abs(cross_products) < tolerance)


class IrregularPolygonEdgeSegment:
    def __init__(self, curviness, curve_location):
        self.curviness = curviness
        self.curve_location = curve_location
        self.points_array = []

    def render(self,centroid, start_point, end_point, colour, thickness, ax_n=None):
        if self.curviness>0:
            t_values = np.linspace(0, 1, line_num_points)
            p0 = start_point
            p2 = end_point
            line_direction = normalised_vector_direction(p0, p2)
            line_direction_angle = np.degrees(np.arctan2(line_direction[1], line_direction[0]))
            p1 = p0 + (self.curve_location * un_normalised_vector_direction(p0,p2)) #moves curve_location away from P0 towards P2 relative to length of curve segment

            to_center_direction = normalised_vector_direction(p1, centroid)
            to_center_direction_angle = np.degrees(np.arctan2(to_center_direction[1], to_center_direction[0]))

            c = (to_center_direction_angle - line_direction_angle) % 360
            #If the direction to the centroid is  +0-180 degrees from the angle of the line then curve 90 else curve -90
            if 0< c <180:
                relative_curve_direction_degrees = line_direction_angle + 90
            elif  c ==0 or c==180:
                relative_curve_direction_degrees = line_direction_angle + 90
                print("aaaaaaaaaaaaaa")
            else:
                relative_curve_direction_degrees = line_direction_angle - 90
            curve_dir_radians = np.radians(relative_curve_direction_degrees)

            # Calculate x and y offsets
            length = math.sqrt((p2[0] - p0[0]) ** 2 + (p2[1] - p0[1]) ** 2)
            dx = length * self.curviness * np.cos(curve_dir_radians)
            dy = length * self.curviness * np.sin(curve_dir_radians)
            p1 = p1 + np.array([dx, dy])
            self.points_array = np.array([bezier_curve_t(t, p0, p1, p2) for t in t_values])
            x_values, y_values = self.points_array[:, 0], self.points_array[:, 1]
        else:
            x_values = np.linspace(start_point[0], end_point[0], line_num_points)
            y_values = np.linspace(start_point[1], end_point[1], line_num_points)
            self.points_array = np.column_stack((x_values, y_values))
            self.points_array = np.round(self.points_array, 3)

        return self.points_array

    def mutate_val(self, value, range, mutation_rate):
        if np.random.random() < mutation_rate:
            mutation_magnitude = mutation_rate * (range[1] - range[0])
            return min(range[1], max(range[0], value * (1 + np.random.normal(-mutation_magnitude, mutation_magnitude))))
        else:
            return value

    def mutate(self, mutation_rate=0.05):
        if self.curviness == 0:
            self.curviness = 0 if np.random.random() > mutation_rate else self.mutate_val(self.curviness, curviness_range,
                                                                                      mutation_rate)
        else:
            self.curviness = self.mutate_val(self.curviness, curviness_range, mutation_rate)

        self.curve_location = self.mutate_val(self.curve_location, relative_location_range, mutation_rate)


class IrregularPolygonSegment(Segment):
    """Line segment with additional properties specific to a line."""
    def __init__(self, segment_type, start, start_mode, end_thickness, relative_angle, colour, bounding_size, corners, lines_list, fill, is_eyeliner_wing=False):
        super().__init__(segment_type, start, start_mode, end_thickness, relative_angle, colour)
        self.segment_type = SegmentType.IRREGULAR_POLYGON
        self.start = start
        self.segment_type = SegmentType.IRREGULAR_POLYGON
        self.bounding_size = bounding_size
        self.corners = corners
        self.lines_list = lines_list
        self.points_array = np.array([])  # Initialize as an empty array
        self.to_scale_corners = np.array([])
        self.is_eyeliner_wing = is_eyeliner_wing
        self.fill = fill


    def rotated_corners(self,corners):
        """Rotates edges around the start by """
        angle_rad = math.radians(self.absolute_angle)
        rotated_corners = []
        ox, oy = self.start  # Origin coordinates (start of shape)
        for corner in corners:
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

        return np.array(rotated_corners)

    def move_corners(self,corners):
        movement = un_normalised_vector_direction(corners[0], self.start)

        for corner in corners:
            corner[0] += movement[0]
            corner[1] += movement[1]
        return corners

    def render(self, prev_array, prev_angle, prev_colour, prev_end_thickness, ax_n=None):
        self.points_array = np.array([])
        if self.start_mode == StartMode.CONNECT and len(prev_array) > 15:
            self.start = (prev_array[-1][0], prev_array[-1][1])
        elif self.start_mode == StartMode.CONNECT and len(prev_array) <= 15:
            end_index = point_in_array(prev_array, 0.5)
            self.start = (prev_array[end_index][0], prev_array[end_index][1])

        if self.start_mode == StartMode.JUMP:
            self.absolute_angle = self.relative_angle
        else:
            self.absolute_angle = (prev_angle + self.relative_angle) % 360

        to_scale_corners = np.array([(corner[0] * self.bounding_size[0], corner[1] * self.bounding_size[1]) for corner in self.corners])
        to_scale_corners= self.rotated_corners(to_scale_corners)
        to_scale_corners = self.move_corners(to_scale_corners)
        self.to_scale_corners = to_scale_corners

        start_point = np.array(to_scale_corners[0])
        n_of_corners = len(self.corners)
        centroid = (
        sum(point[0] for point in to_scale_corners) / n_of_corners, sum(point[1] for point in to_scale_corners) / n_of_corners)
        #if ax_n:
        #    ax_n.scatter(centroid[0], centroid[1], color='red',linewidth=4, zorder=5)

        for i, edge in enumerate(to_scale_corners):
            if i+1 < len(to_scale_corners):
                end_point = np.array(to_scale_corners[i + 1])
                #ax_n.scatter(end_point[0], end_point[1], color='orange', linewidth=3,  zorder=6)
            else:
                end_point = np.array(to_scale_corners[0])
                #ax_n.scatter(end_point[0], end_point[1], color='blue',linewidth=3,  zorder=6)

            section_points_array = self.lines_list[i].render(centroid,start_point, end_point, self.colour, self.end_thickness, ax_n)
            if self.points_array.size == 0:
                self.points_array = section_points_array  # Directly assign if empty
            else:
                self.points_array = np.vstack((self.points_array, section_points_array))
            start_point = end_point

        if ax_n:
            if self.fill == True:
                x, y = self.points_array[:, 0], self.points_array[:, 1]
                plt.fill(x, y, color=self.colour)
            else:
                ax_n.plot(self.points_array[:, 0], self.points_array[:, 1], self.colour, lw=self.end_thickness)  # Plot all points as a single object


    def mutate(self, mutation_rate=0.05):
        if self.start_mode == StartMode.JUMP:
            self.start = (self.mutate_val(self.start[0], start_x_range, mutation_rate),
                          self.mutate_val(self.start[1], start_y_range, mutation_rate))

        self.bounding_size = (self.mutate_val(self.bounding_size[0], random_shape_size_range, mutation_rate),
                          self.mutate_val(self.bounding_size[1], random_shape_size_range, mutation_rate))
        for i , corner in enumerate(self.corners):
            self.corners[i] = (self.mutate_val(corner[0], corner_initialisation_range, mutation_rate),
                          self.mutate_val(corner[1], corner_initialisation_range, mutation_rate))
        for line in self.lines_list:
            line.mutate(mutation_rate)