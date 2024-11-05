from enum import Enum
import math
import matplotlib.pyplot as plt
import numpy as np

import EyelinerWingGeneration
from EyelinerWingGeneration import vector_direction
import StarGeneration
from StarGeneration import *

class SegmentType(Enum):
    LINE = 'LINE'
    #CURVE = 'CURVE'
    #FORK = 'FORK'
    #TAPER = 'TAPER'
    STAR = 'STAR'
    #WING = 'WING'
    #JOIN_LINE = 'JOIN_LINE'


class StartMode(Enum):
    CONNECT = 'CONNECT'
    JUMP = 'JUMP'
    #FORK = 'FORK'

class Segment:
    """Base class for all segments."""
    def __init__(self, start, start_mode, end_thickness):
        self.start = start  # Tuple (x, y)
        self.end = None     # Calculated by subclass
        self.start_mode = start_mode
        self.end_thickness = end_thickness

    def render(self, ax):
        """Base render method to override in subclasses."""
        raise NotImplementedError("Subclasses should implement this!")

    def generate_coordinates(self, prev_end=None):
        """Base method to be overridden by subclasses if needed."""
        if self.start_mode == StartMode.CONNECT and prev_end:
            self.start = prev_end
        return self.start, self.end

class LineSegment(Segment):
    """Line segment with additional properties specific to a line."""
    def __init__(self, start, start_mode, length, direction, start_thickness, end_thickness, color, curviness, curve_direction, curve_location):
        super().__init__(start, start_mode, end_thickness)
        self.length = length
        self.direction = direction  # Angle in degrees
        self.start_thickness = start_thickness
        self.color = color
        self.curviness = curviness
        if curviness>0:  #If curved line then curve direction else no curve direction
            self.curve_direction = curve_direction #curve direction in degrees
            self.curve_location = curve_location
            self.curve_location = curve_location
        else:
            self.curve_direction = 0
            self.curve_location = 0
            self.curve_location =0
        self.calculate_end()  # Calculate the endpoint

    def calculate_end(self):
        """Calculate the end point based on start, length, and direction."""
        radians = math.radians(self.direction)
        end_x = self.start[0] + self.length * math.cos(radians)
        end_y = self.start[1] + self.length * math.sin(radians)
        self.end = (end_x, end_y)

    def render(self, ax):
        """Render a line segment with thickness tapering from start to end."""
        num_steps = 50  # Number of points to create a smooth thickness transition/ curve
        thicknesses = np.linspace(self.start_thickness, self.end_thickness, num_steps)

        if self.curviness>0:
            t_values = np.linspace(0, 1, num_steps)
            P0 = np.array(self.start)
            P2 = np.array(self.end)
            P1 = P0 + ((self.length * self.curve_location) * EyelinerWingGeneration.vector_direction(P0,P2)) #moves curve_location away from P0 towards P2 relative to length of curve segment
            curve_dir_radians = np.radians(self.curve_direction)
            # Calculate x and y offsets
            dx = self.curviness * np.cos(curve_dir_radians)
            dy = self.curviness * np.sin(curve_dir_radians)
            P1 = P1 + np.array([dx, dy])

            curve_points = np.array([bezier_curve(t, P0, P1, P2) for t in t_values])
            print("curve_points shape:", curve_points.shape)
            x_values, y_values = curve_points[:, 0], curve_points[:, 1]
        else:
            x_values = np.linspace(self.start[0], self.end[0], num_steps)
            y_values = np.linspace(self.start[1], self.end[1], num_steps)

        # Plot each small segment with the varying thickness
        for i in range(num_steps - 1):
            ax.plot(
                [x_values[i], x_values[i + 1]], [y_values[i], y_values[i + 1]],
                color=self.color,
                linewidth=thicknesses[i],
                solid_capstyle='round'
            )

    def mutate(self, mutation_rate=0.1):
        """Randomly mutate properties of the line segment within a mutation rate."""
        # Randomly mutate length and direction
        if np.random.uniform(0,1)>0.5:
            self.length *= 1 + np.random.uniform(-mutation_rate, mutation_rate)
        self.direction += np.random.uniform(-mutation_rate,
                                            mutation_rate) * 360  # random angle mutation within ±mutation rate of a full circle

        # Randomly mutate thicknesses
        #self.start_thickness *= 1 + np.random.uniform(-mutation_rate, mutation_rate)
        self.end_thickness *= 1 + np.random.uniform(-mutation_rate, mutation_rate)

        # Randomly mutate curviness, curve direction, and curve location
        if self.curviness > 0:
            self.curviness *= 1 + np.random.uniform(-mutation_rate, mutation_rate)
            self.curve_direction += np.random.uniform(-mutation_rate, mutation_rate) * 360
            self.curve_location = min(1, max(0, self.curve_location + np.random.uniform(-mutation_rate, mutation_rate)))  # keeps curve location within [0, 1]

        # Recalculate the end point
        self.calculate_end()


class StarSegment(Segment):
    """Line segment with additional properties specific to a line."""
    def __init__(self, start, center, radius, arm_length, num_points,asymmetry,curved, start_mode, end_thickness):
        super().__init__(start, start_mode, end_thickness)
        self.center = center
        self.radius = radius
        self.arm_length = arm_length
        self.num_points = num_points
        self.asymmetry = asymmetry
        self.curved = curved
        start_angle = 2 * np.pi * num_points / num_points #angle for last arm
        bin, end_p = StarGeneration.create_star_arm(center, radius,arm_length, num_points,start_angle,asymmetry,num_points,curved) #armN = last arm so num_points
        self.end = end_p

    def render(self, ax):
        star_points,end_coord = StarGeneration.create_star(self.num_points, self.center, self.radius, self.arm_length, self.asymmetry, self.curved)
        plt.plot(star_points[:, 0], star_points[:, 1], 'b', lw=self.end_thickness)  # Plot all points as a single object
        #self.end = end_coord

# Factory function to create a specific segment instance and wrap it in Segment
def create_segment(start, start_mode, segment_type, **kwargs):
    if segment_type == SegmentType.LINE:
        return LineSegment(
            start=start,
            start_mode=start_mode,
            direction=kwargs.get('direction', 0),
            start_thickness=kwargs.get('start_thickness', 1),
            end_thickness=kwargs.get('end_thickness', 1),
            color=kwargs.get('color', 'black'),
            length = kwargs.get('length', 1),
            curviness=kwargs.get('curviness', 0),
            curve_direction=kwargs.get('curve_direction', 0),
            curve_location=kwargs.get('curve_location', 0)
        )
    elif segment_type == SegmentType.STAR:
        return StarSegment(
            start=start,
            start_mode=start_mode,
            center=start, #center is the end of previous object
            radius=kwargs.get('radius', 0.5),
            arm_length=kwargs.get('arm_length', 1),
            num_points=kwargs.get('num_points', 5),
            asymmetry=kwargs.get('asymmetry', 0),
            curved=kwargs.get('curved', True),
            end_thickness = kwargs.get('end_thickness', 1)
        )
    else:
        raise ValueError(f"Unsupported segment type: {segment_type}")


class EyelinerDesign:   #Creates overall design, calculates start points, renders each segment by calling their render function
    def __init__(self):
        self.segments = []

    def add_segment(self, segment):
        self.segments.append(segment)

    def get_next_start_point(self):
        """Choose the next start point based on the last segment's available points."""
        if not self.segments:
            return (0, 0)  # Starting point for the first segment
        last_segment = self.segments[-1]
        return last_segment.end

    def render(self):
        """Render the eyeliner design using matplotlib."""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal', adjustable='box')
        for segment in self.segments:
            segment.render(ax)
        plt.show()

    def get_start_thickness(self):
        if not self.segments:
            return 1 # Starting thickness for the first segment
        last_segment = self.segments[-1]
        return last_segment.end_thickness

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