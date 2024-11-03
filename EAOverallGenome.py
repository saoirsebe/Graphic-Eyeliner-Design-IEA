from enum import Enum
import random
import matplotlib.pyplot as plt

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
    def __init__(self, start, end, start_mode):
        self.start = start  # Tuple (x, y)
        self.end = end      # Tuple (x, y)
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
    def __init__(self, start, end, direction, start_thickness, end_thickness, color, texture, start_mode):
        super().__init__(start, end, start_mode)
        self.direction = direction
        self.start_thickness = start_thickness
        self.end_thickness = end_thickness
        self.color = color
        self.texture = texture

    def render(self, ax):
        """Render a line segment with the specified properties."""
        x_values, y_values = zip(self.start, self.end)
        thickness = (self.start_thickness + self.end_thickness) / 2
        ax.plot(x_values, y_values, color=self.color, linewidth=thickness, linestyle='-')

class CurveSegment(Segment):
    """Curve segment with additional properties specific to a curve."""
    def __init__(self, start, end, curvature, start_thickness, end_thickness, color, texture, start_mode, direction):
        super().__init__(start, end, start_mode)
        self.curvature = curvature
        self.start_thickness = start_thickness
        self.end_thickness = end_thickness
        self.color = color
        self.texture = texture
        self.direction = direction

    def render(self, ax):
        """Render a curved segment with a simple curve approximation."""
        # Generate curved path here; simplified as straight line for this example
        x_values, y_values = zip(self.start, self.end)
        thickness = (self.start_thickness + self.end_thickness) / 2
        ax.plot(x_values, y_values, color=self.color, linewidth=thickness, linestyle='--')

# Factory function to create a specific segment instance and wrap it in Segment
def create_segment(start, end, start_mode, segment_type, **kwargs):
    if segment_type == SegmentType.LINE:
        return LineSegment(
            start=start,
            end=end,
            start_mode=start_mode,
            direction=kwargs.get('direction', 0),
            start_thickness=kwargs.get('start_thickness', 1),
            end_thickness=kwargs.get('end_thickness', 1),
            color=kwargs.get('color', 'black'),
            texture=kwargs.get('texture', 'matte')
        )
    elif segment_type == SegmentType.CURVE:
        return CurveSegment(
            start=start,
            end=end,
            start_mode=start_mode,
            curvature=kwargs.get('curvature', 0),
            start_thickness=kwargs.get('start_thickness', 1),
            end_thickness=kwargs.get('end_thickness', 1),
            color=kwargs.get('color', 'blue'),
            direction=kwargs.get('direction', 0),
            texture=kwargs.get('texture', 'glitter')
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

# Creating a line segment with specific properties and adding it to the design
line_segment = create_segment(
    segment_id=1,
    start=(0, 0),
    end=(1, 2),
    start_mode=StartMode.CONNECT,
    segment_type=SegmentType.LINE,
    direction=45,
    start_thickness=3,
    end_thickness=1,
    color='black',
    texture='matte',
    fork_points=[(0.5, 1), (1, 1.5)]
)
design.add_segment(line_segment)

# Creating a curve segment that connects to the previous segment
next_start = design.get_next_start_point()
curve_segment = create_segment(
    segment_id=2,
    start=next_start,
    end=(2, 3),
    start_mode=StartMode.FORK,
    segment_type=SegmentType.CURVE,
    curvature=0.5,
    start_thickness=2,
    end_thickness=1,
    color='green',
    texture='glitter'
)
design.add_segment(curve_segment)

# Render the design
design.render()
