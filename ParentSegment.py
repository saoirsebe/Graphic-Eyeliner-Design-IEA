import numpy as np
from A import StartMode, colour_options, similar_colours


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
        self.children = []

    def render(self, prev_array, prev_angle, prev_colour, prev_end_thickness, ax_n=None):
        """Base render method to override in subclasses."""
        raise NotImplementedError("Subclasses should implement this!")

    def generate_start(self, prev_end=None):
        """Base method to be overridden by subclasses if needed."""
        if self.start_mode == StartMode.CONNECT and prev_end:
            self.start = prev_end
        return self.start

    def add_child_segment(self, child):
        self.children.append(child)

    def get_children(self):
        return self.children

    def remove_child_segment(self, child):
        self.children.remove(child)

    def mutate(self):
        """Base mutate method to override in subclasses."""
        raise NotImplementedError("Subclasses should implement this!")

    def mutate_val(self, value, range ,mutation_rate):
        if np.random.random() < mutation_rate:
            mutation_magnitude = mutation_rate * (range[1]-range[0])

            # If value is 0, add a normally distributed offset.
            if value == 0:
                new_val = value + abs(np.random.normal(mutation_magnitude, mutation_magnitude))
            # If value is at or below the minimum, force mutation upward.
            elif value <= range[0]:
                new_val = value + abs(np.random.normal(mutation_magnitude, mutation_magnitude))
            # If value is at or above the maximum, force mutation downward.
            elif value >= range[1]:
                new_val = value - abs(np.random.normal(-mutation_magnitude, mutation_magnitude))
            else:
                # For intermediate values, add a percentage-based offset.
                new_val = value * (1 + np.random.normal(0, mutation_magnitude))
                # Ensure new_val is within allowed_range.
            return min(range[1], max(range[0], new_val))
        else:
            return value

    def mutate_colour(self, colour, mutation_rate):
        if not colour or colour not in colour_options:
            colour = self.colour
            print("colour == False in mutate_colour")
        # Decide whether to mutate based on mutation_rate
        if np.random.random() < mutation_rate/2:
            # If a mutation happens, try to pick a similar colour
            similar = similar_colours.get(colour, [])
            # With high probability (e.g., 80%), choose from similar colours if available
            if similar and np.random.random() < 0.8:
                return np.random.choice(similar)
            else:
                # Otherwise, fall back to any random colour from the full set
                return np.random.choice(colour_options)
        else:
            # No mutation, return the original colour
            return colour

    def mutate_choice(self, the_value, options, mutation_rate):
        if np.random.random() < mutation_rate /1.5:
            return np.random.choice(options)
        else:
            return the_value

def point_in_array(array,location_in_array):
    num_points = len(array)
    target_index = round(location_in_array * (num_points - 1))
    #point = array[target_index]
    return target_index
