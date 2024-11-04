from enum import Enum
import math
import matplotlib.pyplot as plt
import numpy as np

import StarGeneration
from StarGeneration import *

class SegmentType(Enum):
    LINE = 'LINE'
    CURVE = 'CURVE'
    #FORK = 'FORK'
    TAPER = 'TAPER'
    STAR = 'STAR'
    WING = 'WING'
    JOIN_LINE = 'JOIN_LINE'


class StartMode(Enum):
    CONNECT = 'CONNECT'
    JUMP = 'JUMP'
    FORK = 'FORK'

class Segment:
    """Base class for all segments."""
    def __init__(self, start, start_mode):
        self.start = start  # Tuple (x, y)
        self.end = None     # Calculated by subclass
        self.start_mode = start_mode

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
    def __init__(self, start, start_mode, length, direction, start_thickness, end_thickness, color, texture):
        super().__init__(start, start_mode)
        self.length = length
        self.direction = direction  # Angle in degrees
        self.start_thickness = start_thickness
        self.end_thickness = end_thickness
        self.color = color
        self.texture = texture
        self.calculate_end()  # Calculate the endpoint

    def calculate_end(self):
        """Calculate the end point based on start, length, and direction."""
        radians = math.radians(self.direction)
        end_x = self.start[0] + self.length * math.cos(radians)
        end_y = self.start[1] + self.length * math.sin(radians)
        self.end = (end_x, end_y)

    def render(self, ax):
        """Render a line segment with thickness tapering from start to end."""
        num_steps = 50  # Number of points to create a smooth thickness transition
        x_values = np.linspace(self.start[0], self.end[0], num_steps)
        y_values = np.linspace(self.start[1], self.end[1], num_steps)
        thicknesses = np.linspace(self.start_thickness, self.end_thickness, num_steps)

        # Plot each small segment with the varying thickness
        for i in range(num_steps - 1):
            ax.plot(
                [x_values[i], x_values[i + 1]], [y_values[i], y_values[i + 1]],
                color=self.color,
                linewidth=thicknesses[i],
                solid_capstyle='round'
            )

class CurveSegment(Segment):
    """Curve segment with additional properties specific to a curve."""
    def __init__(self, start, start_mode, length, curvature, start_thickness, end_thickness, color, texture, direction):
        super().__init__(start, start_mode)
        self.length = length
        self.curvature = curvature
        self.start_thickness = start_thickness
        self.end_thickness = end_thickness
        self.color = color
        self.texture = texture
        self.direction = direction  # Angle in degrees
        self.calculate_end()  # Calculate the endpoint

    def calculate_end(self):
        """Calculate a simple curved endpoint based on start and length."""
        # This is a basic example; curvature could define a variety of complex paths
        # Assuming curvature modifies direction within a small range, for simplicity
        direction_angle = 45  # Fixed angle for simplicity in this example
        radians = math.radians(direction_angle)
        end_x = self.start[0] + self.length * math.cos(radians)
        end_y = self.start[1] + self.length * math.sin(radians)
        self.end = (end_x, end_y)

    def render(self, ax):
        """Render a curved segment with a simple curve approximation."""
        # Generate curved path here; simplified as straight line for this example
        x_values, y_values = zip(self.start, self.end)
        thickness = (self.start_thickness + self.end_thickness) / 2
        ax.plot(x_values, y_values, color=self.color, linewidth=thickness, linestyle='--')

class StarSegment(Segment):
    """Line segment with additional properties specific to a line."""
    def __init__(self, start, center, radius, arm_length, num_points,asymmetry,curved, start_mode):
        super().__init__(start, start_mode)
        self.center = center
        self.radius = radius
        self.arm_length = arm_length
        self.num_points = num_points
        self.asymmetry = asymmetry
        self.curved = curved
        start_angle = 2 * np.pi * num_points / num_points #angle for last arm
        bin, end_p = StarGeneration.create_star_arm(center, radius, num_points,arm_length,start_angle,asymmetry,num_points,curved) #armN = last arm so num_points
        self.end = end_p

    def render(self, ax):
        star_points,end_coord = StarGeneration.create_star(self.num_points, self.center, self.radius, self.arm_length, self.asymmetry, self.curved)
        plt.plot(star_points[:, 0], star_points[:, 1], 'b', lw=2)  # Plot all points as a single object
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
            texture=kwargs.get('texture', 'matte'),
            length = kwargs.get('length', 1)
        )
    elif segment_type == SegmentType.CURVE:
        return CurveSegment(
            start=start,
            start_mode=start_mode,
            curvature=kwargs.get('curvature', 0),
            start_thickness=kwargs.get('start_thickness', 1),
            end_thickness=kwargs.get('end_thickness', 1),
            color=kwargs.get('color', 'blue'),
            direction=kwargs.get('direction', 0),
            texture=kwargs.get('texture', 'glitter'),
            length=kwargs.get('length', 1),
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
        )
    else:
        raise ValueError(f"Unsupported segment type: {segment_type}")



class EyelinerDesign:
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

# Example usage
design = EyelinerDesign()

# Adding a line segment to the design
line_segment = create_segment(
    start=(0, 0),
    start_mode=StartMode.CONNECT,
    segment_type=SegmentType.LINE,
    length=2,
    direction=45,
    start_thickness=5,
    end_thickness=1,
    color='black',
    texture='matte'
)
design.add_segment(line_segment)

# Adding a curve segment that connects to the previous segment
next_start = design.get_next_start_point()
star_segment = create_segment(
    start=next_start,
    start_mode=StartMode.CONNECT,
    segment_type=SegmentType.STAR,
    radius=0.5,
    arm_length=1,
    num_points=4,
    asymmetry=0.5,
    curved=False
)
design.add_segment(star_segment)

next_start = design.get_next_start_point()
curve_segment = create_segment(
    start=next_start,
    start_mode=StartMode.CONNECT,
    segment_type=SegmentType.CURVE,
    length=1.5,
    direction=20,
    curvature=0.5,
    start_thickness=2,
    end_thickness=1,
    color='green',
    texture='glitter'
)
design.add_segment(curve_segment)

# Render the design
design.render()
