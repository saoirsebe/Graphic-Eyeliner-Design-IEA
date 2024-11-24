from enum import Enum
import math
import matplotlib.pyplot as plt
import numpy as np

import EyelinerWingGeneration
from EyelinerWingGeneration import vector_direction, draw_eye_shape
import StarGeneration
from StarGeneration import *
from Trial import start_x_range, start_y_range, thickness_range, direction_range, length_range, arm_length_range, \
    num_points_range, asymmetry_range, curviness_range, relative_location_range, radius_range

colour_options = [
    "red", "green", "blue", "cyan", "magenta", "black", "orange", "purple",
    "pink", "brown", "gray", "lime", "navy",
    "teal", "maroon", "gold", "olive"
]


class SegmentType(Enum):
    LINE = 'LINE'
    #FORK = 'FORK'
    #TAPER = 'TAPER'
    STAR = 'STAR'
    #WING = 'WING'

class StartMode(Enum):
    CONNECT = 'CONNECT'
    JUMP = 'JUMP'
    #FORK = 'FORK'
    SPLIT = 'SPLIT' #connect but end of prev is in segment
    CONNECT_MID = 'CONNECT_MID'


class Segment:
    """Base class for all segments."""
    def __init__(self, segment_type, start, start_mode, end_thickness, relative_angle, colour):
        self.segment_type = segment_type
        self.start = start  # Tuple (x, y)
        self.start_mode = start_mode
        self.end_thickness = end_thickness
        self.relative_angle = relative_angle
        self.absolute_angle = 0  # initialise to 0 and re-set after rendered
        self.points_array = []
        self.colour = colour

    def render(self, ax_n, prev_array, prev_angle, prev_colour, prev_end_thickness):
        """Base render method to override in subclasses."""
        raise NotImplementedError("Subclasses should implement this!")

    def generate_start(self, prev_end=None):
        """Base method to be overridden by subclasses if needed."""
        if self.start_mode == StartMode.CONNECT and prev_end:
            self.start = prev_end
        return self.start

    def mutate(self):
        """Base mutate method to override in subclasses."""
        raise NotImplementedError("Subclasses should implement this!")

    def mutate_val(self,value,range,mutation_rate):
        return min(range[1], max(range[0], value * (1 + np.random.normal(-mutation_rate, mutation_rate))))

def point_in_array(array,location_in_array):
    num_points = len(array)
    target_index = round(location_in_array * (num_points - 1))
    point = array[target_index]
    return target_index

class LineSegment(Segment):
    """Line segment with additional properties specific to a line."""
    def __init__(self, segment_type, start, start_mode, length, relative_angle, start_thickness, end_thickness, colour, curviness, curve_direction, curve_location, start_location, split_point):
        super().__init__(segment_type, start, start_mode, end_thickness, relative_angle, colour)
        self.end = None  # Calculated in render
        self.length = length
        self.start_thickness = start_thickness
        self.curviness = curviness
        if curviness>0:  #If curved line then curve direction else no curve direction
            self.curve_direction = curve_direction #curve direction in degrees
            self.curve_location = curve_location
            self.curve_location = curve_location
        else:
            self.curve_direction = 0
            self.curve_location = 0
        if start_mode == StartMode.CONNECT_MID:
            self.start_location = start_location
            self.split_point = 0
        elif start_mode == StartMode.SPLIT:
            self.split_point = split_point  #if mode is split then set split_point else splits from 0 (i.e. no split)
            self.start_location = 1
        else:
            self.start_location = 1
            self.split_point = 0


    def calculate_end(self, prev_angle):
        """Calculate the end point based on start, length, and direction."""
        self.absolute_angle = prev_angle + self.relative_angle

        radians = math.radians(self.absolute_angle) # Convert to radians
        end_x = self.start[0] + self.length * math.cos(radians)
        end_y = self.start[1] + self.length * math.sin(radians)
        self.end = (end_x, end_y)


    def curve_between_lines(self, P0, P1, P2, P3, colour):
        t_values = np.linspace(0, 1, 20)
        left_curve = np.array([bezier_curve(t, P2, P1, P0) for t in t_values])
        # np.array((prev_array[start_array_point_index+2]) - np.array(self.points_array[2]))/2
        right_curve = np.array([bezier_curve(t, P3, P1, P2) for t in t_values])
        left_x, left_y = left_curve[:, 0], left_curve[:, 1]
        right_x, right_y = right_curve[:, 0], right_curve[:, 1]

        # Combine the points to form a closed boundary
        boundary_x = np.concatenate([left_x, right_x[::1]])  # Combine left and reversed right x
        boundary_y = np.concatenate([left_y, right_y[::1]])  # Combine left and reversed right y
        plt.fill(boundary_x, boundary_y, color=colour, alpha=0.5)
        return np.concatenate((left_curve, right_curve, self.points_array), axis=0)

    def render(self, ax_n, prev_array, prev_angle, prev_colour, prev_end_thickness):
        self.absolute_angle = prev_angle + self.relative_angle
        new_array = []
        num_steps = 50  # Number of points to create a smooth thickness transition/ curve
        if self.start_mode == StartMode.CONNECT and len(prev_array)>15 or self.start_mode == StartMode.SPLIT and len(prev_array)>15:
            self.start = (prev_array[-1][0], prev_array[-1][1])
            self.start_thickness = prev_end_thickness
        elif self.start_mode == StartMode.CONNECT and len(prev_array)<=15 or self.start_mode == StartMode.SPLIT and len(prev_array)<=15:
            end_index = point_in_array(prev_array, 0.5)
            self.start = (prev_array[end_index][0], prev_array[end_index][1])
        elif self.start_mode == StartMode.CONNECT_MID and len(prev_array)>0:
            start_array_point_index = point_in_array(prev_array, self.start_location)
            start_array_point = prev_array[start_array_point_index]
            self.start = (start_array_point[0], start_array_point[1])

        self.calculate_end(prev_angle)  # Calculate the endpoint

        if self.curviness>0:
            t_values = np.linspace(0, 1, num_steps)
            P0 = np.array(self.start)
            P2 = np.array(self.end)
            P1 = P0 + ((self.length * self.curve_location) * EyelinerWingGeneration.vector_direction(P0,P2)) #moves curve_location away from P0 towards P2 relative to length of curve segment
            relative_curve_direction = self.absolute_angle + self.curve_direction
            curve_dir_radians = np.radians(relative_curve_direction)
            # Calculate x and y offsets
            dx = self.curviness * np.cos(curve_dir_radians)
            dy = self.curviness * np.sin(curve_dir_radians)
            P1 = P1 + np.array([dx, dy])

            self.points_array = np.array([bezier_curve(t, P0, P1, P2) for t in t_values])
            x_values, y_values = self.points_array[:, 0], self.points_array[:, 1]
        else:
            x_values = np.linspace(self.start[0], self.end[0], num_steps)
            y_values = np.linspace(self.start[1], self.end[1], num_steps)
            self.points_array = np.column_stack((x_values, y_values))

        """Set self.points_array and move line if split type"""
        if self.start_mode == StartMode.SPLIT:
            split_point_point_index = point_in_array(self.points_array, self.split_point)
            split_point_point = self.points_array[split_point_point_index]
            transformation_vector = vector_direction(split_point_point,self.start)

            self.points_array = np.array([(point[0] + transformation_vector[0], point[1] + transformation_vector[1]) for point in self.points_array])
            x_values, y_values = self.points_array[:, 0], self.points_array[:, 1]

            """Add curve from prev line to split point"""
            if len(prev_array) > 20:
                P2 = np.array(prev_array[-10])
                if split_point_point_index <= 5:
                    beep = 0
                else:
                    beep = split_point_point_index - 5
                P0 = np.array(self.points_array[beep])
                if len(self.points_array) <= (split_point_point_index + 5):
                    beep = len(prev_array) - 1
                else:
                    beep = split_point_point_index + 5
                P3 = np.array(self.points_array[beep])
                P1 = np.array(self.start)
                new_array = self.curve_between_lines(P0, P1, P2, P3, prev_colour)
                x_values, y_values = new_array[:, 0], new_array[:, 1]

        thicknesses = np.linspace(self.start_thickness, self.end_thickness, num_steps) #Render a line segment with thickness tapering from start to end
        """Add curve from bottom of connect mid line"""
        if self.start_mode == StartMode.CONNECT_MID and len(prev_array)>20:
            P2 = np.array(self.points_array[10])
            if start_array_point_index <= 5:
                beep = 0
            else:
                beep = start_array_point_index - 5
            P0 = np.array(prev_array[beep])
            if len(prev_array) <= (start_array_point_index + 5):
                beep = len(prev_array) - 1
            else:
                beep = start_array_point_index + 5
            P3 = np.array(prev_array[beep])
            P1 = np.array(self.start)
            new_array = self.curve_between_lines(P0, P1, P2, P3,self.colour)
            x_values, y_values = new_array[:, 0], new_array[:, 1]

        # Plot each small segment with the varying thickness
        if len(new_array) > 0:
            # Plot the first 40 points as one segment
            for i in range(min(39, len(x_values) - 1)):
                this_colour = self.colour
                if self.start_mode == StartMode.CONNECT_MID:
                    thickness = thicknesses[10]
                elif self.start_mode == StartMode.SPLIT:
                    thickness = thicknesses[0]
                    this_colour = prev_colour
                else:
                    thickness = thicknesses[i]
                ax_n.plot(
                    [x_values[i], x_values[i + 1]], [y_values[i], y_values[i + 1]],
                    color=this_colour,
                    linewidth=thickness,
                    solid_capstyle='butt',
                )

            # Plot remaining points
            for i in range(40, len(x_values) - 1):
                if self.start_mode == StartMode.CONNECT_MID or self.start_mode == StartMode.SPLIT:
                    thickness = thicknesses[i - 41]
                else:
                    thickness = thicknesses[i]
                ax_n.plot(
                    [x_values[i], x_values[i + 1]], [y_values[i], y_values[i + 1]],
                    color=self.colour,
                    linewidth=thickness,
                    solid_capstyle='butt',
                )
        else:
            # Plot normally if no new_array exists (Start type is jump or connect so no blend between lines needed)
            for i in range(len(x_values) - 1):
                thickness = thicknesses[i]
                ax_n.plot(
                    [x_values[i], x_values[i + 1]], [y_values[i], y_values[i + 1]],
                    color=self.colour,
                    linewidth=thickness,
                    solid_capstyle='butt',
                )


    def mutate(self, mutation_rate=0.1):
        """Randomly mutate properties of the line segment within a mutation rate."""
        # mutation rate chance of changing:
        if np.random.normal() < mutation_rate:
            self.start_mode = np.random.choice([StartMode.CONNECT, StartMode.JUMP, StartMode.SPLIT, StartMode.CONNECT_MID])
        if self.start_mode == StartMode.JUMP:
            self.start = (self.mutate_val(self.start[0],start_x_range,mutation_rate),self.mutate_val(self.start[1],start_y_range, mutation_rate))

        if self.start_mode == StartMode.CONNECT_MID:
            self.start_location = self.mutate_val(self.start_location,relative_location_range,mutation_rate)
            self.split_point = 0
        elif self.start_mode == StartMode.SPLIT:
            self.split_point = self.mutate_val(self.split_point, relative_location_range, mutation_rate)
            self.start_location = 1
        else:
            self.start_location = 1
            self.split_point = 0

        self.end_thickness = self.mutate_val(self.end_thickness,thickness_range,mutation_rate)
        self.relative_angle =  self.mutate_val(self.relative_angle,direction_range,mutation_rate)
        # mutation rate chance of changing:
        if np.random.normal() < mutation_rate:
            self.colour = np.random.choice(colour_options)

        self.length = self.mutate_val(self.length,length_range,mutation_rate)
        self.start_thickness = self.mutate_val(self.start_thickness, thickness_range, mutation_rate)

        if self.curviness ==0:
            self.curviness = 0 if np.random.normal() > mutation_rate else self.mutate_val(self.curviness,curviness_range,mutation_rate)
        else:
            self.curviness = self.mutate_val(self.curviness,curviness_range,mutation_rate)

        if self.curviness > 0:
            self.curve_direction = self.mutate_val(self.curve_direction,direction_range,mutation_rate)
            self.curve_location = self.mutate_val(self.curve_location,relative_location_range,mutation_rate)
        else:
            self.curve_direction = 0
            self.curve_location = 0


class StarSegment(Segment):
    """Line segment with additional properties specific to a line."""
    def __init__(self, segment_type, start, colour, center, radius, arm_length, num_points,asymmetry,curved, start_mode, end_thickness, relative_angle):
        super().__init__(segment_type, start, start_mode, end_thickness,relative_angle, colour)
        self.center = center
        self.radius = radius
        self.arm_length = arm_length
        self.num_points = num_points
        self.asymmetry = asymmetry
        self.curved = curved
        #start_angle = 2 * np.pi * num_points / num_points #angle for last arm
        #bin, end_p = StarGeneration.create_star_arm(center, radius,arm_length, num_points,start_angle,asymmetry,num_points,curved) #armN = last arm so num_points
        self.arm_points_array = []


    def render(self, ax_n, prev_array, prev_angle, prev_colour, prev_end_thickness):
        if self.start_mode == StartMode.CONNECT and len(prev_array)>15:
            self.start = (prev_array[-1][0], prev_array[-1][1])
            self.center = self.start
        elif self.start_mode == StartMode.CONNECT and len(prev_array)<=15:
            end_index = point_in_array(prev_array, 0.5)
            self.start = (prev_array[end_index][0], prev_array[end_index][1])
            self.center = self.start
        #elif self.start_mode == StartMode.CONNECT_MID and prev_array:

        self.absolute_angle = prev_angle + self.relative_angle

        star_points, star_arm_points = StarGeneration.create_star(self.num_points, self.center, self.radius, self.arm_length, self.asymmetry, self.curved, self.absolute_angle)
        start_coord = star_arm_points[-1]
        transformation_vector = (self.center[0] - start_coord[0], self.center[1] - start_coord[1])
        #self.end = (end_coord[0]+transformation_vector[0], end_coord[1]+transformation_vector[1])
        transformed_star_points = np.array([(point[0] + transformation_vector[0], point[1] + transformation_vector[1]) for point in star_points])
        ax_n.plot(transformed_star_points[:, 0], transformed_star_points[:, 1], self.colour, lw=self.end_thickness)  # Plot all points as a single object
        transformed_arm_points = np.array([(point[0] + transformation_vector[0], point[1] + transformation_vector[1]) for point in star_arm_points])
        self.arm_points_array = transformed_arm_points
        self.points_array = transformed_star_points

    def mutate(self,mutation_rate=0.1):
        # mutation rate chance of changing:
        if np.random.normal() < mutation_rate:
            self.start_mode = np.random.choice(
                [StartMode.CONNECT, StartMode.JUMP])
        if self.start_mode == StartMode.JUMP:
            self.start = (self.mutate_val(self.start[0], start_x_range, mutation_rate),
                          self.mutate_val(self.start[1], start_y_range, mutation_rate))

        self.end_thickness = self.mutate_val(self.end_thickness, thickness_range, mutation_rate)
        self.relative_angle = self.mutate_val(self.relative_angle, direction_range, mutation_rate)
        # mutation rate chance of changing:
        if np.random.normal() < mutation_rate:
            self.colour = np.random.choice(colour_options)

        self.radius = self.mutate_val(self.radius,radius_range, mutation_rate)
        self.arm_length = self.mutate_val(self.arm_length,arm_length_range, mutation_rate)
        self.num_points = self.mutate_val(self.num_points,num_points_range, mutation_rate)
        self.asymmetry = self.mutate_val(self.asymmetry,asymmetry_range, mutation_rate)
        # mutation rate chance of changing:
        if self.curved:
            self.curved = False if np.random.normal() < mutation_rate else True
        else:
            self.curved = True if np.random.normal() < mutation_rate else False


# Factory function to create a specific segment instance and wrap it in Segment
def create_segment(start, start_mode, segment_type, **kwargs):
    if segment_type == SegmentType.LINE:
        return LineSegment(
            segment_type = segment_type,
            start=start,
            start_mode=start_mode,
            relative_angle=kwargs.get('relative_angle', 0),
            start_thickness=kwargs.get('start_thickness', 1),
            end_thickness=kwargs.get('end_thickness', 1),
            colour=kwargs.get('colour', 'black'),
            length = kwargs.get('length', 1),
            curviness=kwargs.get('curviness', 0),
            curve_direction=kwargs.get('curve_direction', 90),
            curve_location=kwargs.get('curve_location', 0.5),
            start_location = kwargs.get('start_location', 1),
            split_point = kwargs.get('split_point', 0.5)

        )
    elif segment_type == SegmentType.STAR:
        return StarSegment(
            segment_type=segment_type,
            start=start,
            start_mode=start_mode,
            center=start, #center is the end of previous object
            radius=kwargs.get('radius', 0.5),
            arm_length=kwargs.get('arm_length', 1),
            num_points=kwargs.get('num_points', 5),
            asymmetry=kwargs.get('asymmetry', 0),
            curved=kwargs.get('curved', True),
            end_thickness = kwargs.get('end_thickness', 1),
            relative_angle =kwargs.get('relative_angle', 0),
            colour=kwargs.get('colour', 'black')
        )
    else:
        raise ValueError(f"Unsupported segment type: {segment_type}")


class EyelinerDesign:   #Creates overall design, calculates start points, renders each segment by calling their render function
    def __init__(self):
        self.segments = []
        self.n_of_lines= 0
        self.n_of_stars= 0
        self.n_of_segments= 0

    def add_segment(self, segment):
        self.segments.append(segment)

    def render(self):
        fig, ax_n = plt.subplots(figsize=(3, 3))
        """Render the eyeliner design using matplotlib."""
        draw_eye_shape(ax_n)
        segment_n = 0
        prev_array = np.array([self.segments[0].start])
        prev_angle = 0
        prev_colour = self.segments[0].colour
        prev_end_thickness = self.segments[0].end_thickness
        for segment in self.segments:
            segment.render(ax_n, prev_array, prev_angle,prev_colour,prev_end_thickness)
            if segment.segment_type == SegmentType.STAR:
                prev_array = self.segments[segment_n].arm_points_array #if previous segment was a star then pass in the arm points that the next segment should start at
            else:
                prev_array = self.segments[segment_n].points_array
            prev_angle = self.segments[segment_n].absolute_angle
            prev_colour = self.segments[segment_n].colour
            prev_end_thickness = self.segments[segment_n].end_thickness
            segment_n += 1
        return fig

    def get_start_thickness(self):
        if not self.segments:
            return 1 # Starting thickness for the first segment
        last_segment = self.segments[-1]
        return last_segment.end_thickness

    def update_design_info(self):
        self.n_of_lines = 0
        self.n_of_stars = 0
        self.n_of_segments = 0
        for segment in self.segments:
            self.n_of_segments += 1
            if segment.segment_type == SegmentType.LINE:
                self.n_of_lines += 1
            elif segment.segment_type == SegmentType.STAR:
                self.n_of_stars += 1

"""
# Example usage
design = EyelinerDesign()

# Adding a line segment to the design
line_segment = create_segment(
    start=(0, 0),
    start_mode=StartMode.CONNECT,
    segment_type=SegmentType.LINE,
    length=2,
    direction=45,
    start_thickness=10,
    end_thickness=1,
    color='black',
    texture='matte'
)

line_segment.mutate(0.1)
design.add_segment(line_segment)

next_start_thickness = design.get_start_thickness()
next_start = design.get_next_start_point()
curved_line_segment = create_segment(
    start=next_start,
    start_mode=StartMode.CONNECT,
    segment_type=SegmentType.LINE,
    length=1.5,
    direction=20,
    curvature=0.5,
    start_thickness=next_start_thickness,
    end_thickness=1,
    color='green',
    curviness=1,
    curve_direction=-85,
    curve_location=0.8
)
design.add_segment(curved_line_segment)

next_start = design.get_next_start_point()
next_start_thickness = design.get_start_thickness()
star_segment = create_segment(
    start=next_start,
    start_mode=StartMode.CONNECT,
    segment_type=SegmentType.STAR,
    radius=0.5,
    arm_length=1,
    num_points=4,
    asymmetry=0.5,
    curved=True,
    end_thickness=next_start_thickness
)
design.add_segment(star_segment)

# Render the design
design.render()
"""