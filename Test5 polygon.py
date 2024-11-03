from enum import Enum
import random
import matplotlib.pyplot as plt
import numpy as np

class SegmentType(Enum):
    LINE = 'LINE'
    CURVE = 'CURVE'
    SPLIT = 'SPLIT'
    FORK = 'FORK'
    TAPER = 'TAPER'
    STAR = 'STAR'

class StartMode(Enum):
    CONNECT = 'CONNECT'
    JUMP = 'JUMP'
    OVERLAY = 'OVERLAY'

class Segment:
    def __init__(self, segment_id, segment_type, start_mode, start, end, direction, curvature,
                 start_thickness, end_thickness, color, texture, branch_count=0,
                 branch_angle_variation=0, effect_overlay=False):
        self.segment_id = segment_id
        self.segment_type = segment_type
        self.start_mode = start_mode
        self.start = start  # Tuple (x, y)
        self.end = end  # Tuple (x, y)
        self.direction = direction
        self.curvature = curvature
        self.start_thickness = start_thickness
        self.end_thickness = end_thickness
        self.color = color
        self.texture = texture
        self.branch_count = branch_count
        self.branch_angle_variation = branch_angle_variation
        self.effect_overlay = effect_overlay


    def mutate(self):
        #Apply a small random mutation to segment properties for evolutionary variability.
        if random.random() < 0.2:  # 20% chance to mutate curvature
            self.curvature += random.uniform(-0.2, 0.2)
        if random.random() < 0.2:  # 20% chance to mutate thickness
            self.start_thickness = max(0.1, self.start_thickness + random.uniform(-0.5, 0.5))
            self.end_thickness = max(0.1, self.end_thickness + random.uniform(-0.5, 0.5))
        if random.random() < 0.1:  # 10% chance to change direction slightly
            self.direction += random.uniform(-10, 10)

    def generate_coordinates(self, prev_end=None):
        """Generates the coordinates based on segment type and curvature."""
        if self.start_mode == StartMode.CONNECT and prev_end:
            self.start = prev_end
        # Use a basic model for coordinates; more complex shapes need more parameters
        # Here, simulate a basic line/curve segment based on direction, curvature, and thickness
        # Placeholder for calculation based on self.direction, self.curvature, etc.
        return self.start, self.end


class EyelinerDesign:
    def __init__(self):
        self.segments = []

    def add_segment(self, segment):
        self.segments.append(segment)

    def mutate(self):
        """Mutate each segment slightly to create a new variation."""
        for segment in self.segments:
            segment.mutate()

    def render(self):
        """Render the eyeliner design using matplotlib."""
        plt.figure(figsize=(6, 6))
        prev_end = None
        for segment in self.segments:
            start, end = segment.generate_coordinates(prev_end)
            prev_end = end

            # Render each segment with thickness variation
            x_values, y_values = zip(start, end)
            thickness = (segment.start_thickness + segment.end_thickness) / 2

            plt.plot(x_values, y_values, color=segment.color, linewidth=thickness,
                     alpha=0.8 if segment.effect_overlay else 1.0, linestyle='-')

        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


# Example genome setup and rendering
design = EyelinerDesign()
design.add_segment(Segment(
    segment_id=1, segment_type=SegmentType.LINE, start_mode=StartMode.JUMP,
    start=(0, 0), end=(1, 2), direction=45, curvature=0,
    start_thickness=3, end_thickness=1, color='black', texture='matte', effect_overlay=True
))
design.add_segment(Segment(
    segment_id=2, segment_type=SegmentType.CURVE, start_mode=StartMode.CONNECT,
    start=None, end=(2, 3), direction=30, curvature=0.5,
    start_thickness=1, end_thickness=1, color='green', texture='glitter', effect_overlay=False
))
design.add_segment(Segment(
    segment_id=3, segment_type=SegmentType.FORK, start_mode=StartMode.CONNECT,
    start=None, end=(3, 4), direction=15, curvature=0,
    start_thickness=1, end_thickness=0.5, color='black', texture='matte',
    branch_count=3, branch_angle_variation=20, effect_overlay=True
))

# Render the design
design.render()

# Mutate the design and render again to see evolution in action
design.mutate()
design.render()

