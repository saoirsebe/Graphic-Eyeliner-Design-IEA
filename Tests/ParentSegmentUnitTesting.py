import numpy as np
import pytest
import math
from Segments import Segment, point_in_array
from A import SegmentType, StartMode, colour_options, similar_colours

# --- Dummy subclass for testing the base methods ---
class DummySegment(Segment):
    def __init__(self, segment_type, start, start_mode, end_thickness, relative_angle, colour):
        super().__init__(segment_type, start, start_mode, end_thickness, relative_angle, colour)
    def render(self, prev_array, prev_angle, prev_colour, prev_end_thickness, ax_n=None):
        pass  # Not needed for these tests.
    def mutate(self):
        raise NotImplementedError("Not testing mutate() here.")

# --- Test generate_start ---
@pytest.mark.parametrize("start_mode, prev_end, expected_start", [
    # If start_mode is CONNECT and prev_end is provided, start should update.
    (StartMode.CONNECT, (10, 10), (10, 10)),
    # Otherwise, should remain unchanged.
    (StartMode.JUMP, (10, 10), (0, 0)),
    (StartMode.CONNECT_MID, (10, 10), (0, 0)),  # Since base method doesn't change for CONNECT_MID.
])
def test_generate_start(start_mode, prev_end, expected_start):
    dummy = DummySegment(SegmentType.LINE, (0, 0), start_mode, 1, 45, "blue")
    result = dummy.generate_start(prev_end)
    assert result == expected_start, f"Expected start {expected_start}, got {result}"

# --- Test children management methods ---
def test_children_management():
    dummy = DummySegment(SegmentType.LINE, (0, 0), StartMode.CONNECT, 1, 45, "blue")
    # Initially, children should be empty.
    assert dummy.get_children() == [], "Expected empty children list."
    # Add a child.
    child = DummySegment(SegmentType.LINE, (1, 1), StartMode.JUMP, 1, 30, "red")
    dummy.add_child_segment(child)
    assert dummy.get_children() == [child], "Child not added properly."
    # Remove the child.
    dummy.remove_child_segment(child)
    assert dummy.get_children() == [], "Child not removed properly."

# --- Test mutate_val ---
@pytest.mark.parametrize("value, value_range, mutation_rate, fixed_random, fixed_normal, expected", [
    # If random.random() >= mutation_rate, value should remain unchanged.
    (10, (0, 20), 0.1, 0.5, 0, 10),
    # mutation occurs, we simulate a normal deviation of 0 (i.e. no change) so value remains unchanged.
    (10, (0, 20), 1.0, 0.0, 0, 10),
    # np.random.normal returns 0.1, so value becomes 10 * (1+0.1)=11
    (10, (0, 20), 1.0, 0.0, 0.1, 11),
    # np.random.normal returns 0.2, so value becomes 19 * (1+0.2)=22.8 (but bounded by max)
    (19, (0, 20), 1.0, 0.0, 0.2, 20),
    # np.random.normal returns 0.2 and value is 0, so value becomes 0.2
    (0, (0, 20), 1.0, 0.0, 0.2, 0.2),
    # np.random.normal returns 0.2 and value is max of range, so value becomes 20 - 0.2 = 19.8
    (20, (0, 20), 1.0, 0.0, 0.2, 19.8),
])
def test_mutate_val(monkeypatch, value, value_range, mutation_rate, fixed_random, fixed_normal, expected):
    # Monkeypatch np.random.random and np.random.normal.
    monkeypatch.setattr(np.random, "random", lambda: fixed_random)
    monkeypatch.setattr(np.random, "normal", lambda mu, sigma: fixed_normal)
    dummy = DummySegment(SegmentType.LINE, (0,0), StartMode.CONNECT, 1, 45, "blue")
    result = dummy.mutate_val(value, value_range, mutation_rate)
    assert math.isclose(result, expected, rel_tol=1e-5), f"Expected mutate_val to return {expected}, got {result}"

# --- Test mutate_colour ---
@pytest.mark.parametrize("input_colour, mutation_rate, fixed_rand1, fixed_rand2, expected", [
    # If random.random() is high, no mutation occurs.
    ("blue", 0.1, 0.9, 0.9, ["blue"]),
    # If mutation occurs, choose from similar options.
    ("blue", 1.0, 0.0, 0.5, ["green", "indigo", "lightblue", "navy", "darkblue", "cyan", "black"]),
    # If mutation occurs, choose from similar options.
    ("green", 1.0, 0.0, 0.6, ["yellow", "blue", "cyan", "olive", "darkgreen", "teal", "black"]),
    # If mutation occurs but second random.random() is high, choose from full options.
    ("green", 1.0, 0.0, 0.9, ["red", "orange", "yellow", "green", "blue", "indigo", "violet",
    "lightblue", "pink", "purple", "brown", "grey", "black", "cyan", "magenta",
    "maroon", "navy", "olive", "teal", "darkred", "darkgreen", "darkblue",
    "darkorange", "darkviolet", "darkgrey"]),
])
def test_mutate_colour(monkeypatch, input_colour, mutation_rate, fixed_rand1, fixed_rand2, expected):
    # Prepare two fixed random values using a list.
    values = [fixed_rand1, fixed_rand2]
    monkeypatch.setattr(np.random, "random", lambda: values.pop(0))

    # Create a dummy segment instance.
    dummy = DummySegment(SegmentType.LINE, (0, 0), StartMode.CONNECT, 1, 45, input_colour)

    # Call mutate_colour.
    result = dummy.mutate_colour(input_colour, mutation_rate)
    # Check that the result is in the expected list.
    assert result in expected, f"Expected result to be one of {expected}, got {result}"

# --- Test mutate_choice ---
@pytest.mark.parametrize("the_value, options, mutation_rate, fixed_rand, expected", [
    # If mutation does not occur, return the original.
    (5, [1, 2, 5], 0.1, 0.9, 5),
    # If mutation occurs, return a choice from options.
    (5, [1, 2, 5], 1.0, 0.0, 1)  # our fake choice (we'll force np.random.choice to return first element)
])
def test_mutate_choice(monkeypatch, the_value, options, mutation_rate, fixed_rand, expected):
    monkeypatch.setattr(np.random, "random", lambda: fixed_rand)
    monkeypatch.setattr(np.random, "choice", lambda opts: opts[0])
    dummy = DummySegment(SegmentType.LINE, (0,0), StartMode.CONNECT, 1, 45, "blue")
    result = dummy.mutate_choice(the_value, options, mutation_rate)
    assert result == expected, f"Expected mutate_choice to return {expected}, got {result}"

# --- Test for point_in_array ---
@pytest.mark.parametrize("array_length, location_in_array, expected_index", [
    (10, 0.0, 0),
    (10, 0.5, 4),
    (10, 1.0, 9),
    (5, 0.2, 1)
])
def test_point_in_array(array_length, location_in_array, expected_index):
    arr = list(range(array_length))
    result = point_in_array(arr, location_in_array)
    assert result == expected_index, f"Expected index {expected_index}, got {result}"