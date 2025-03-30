import math
import numpy as np
import pytest
import random
import matplotlib.pyplot as plt

# Import the LineSegment class and helper constant from your production module.
from Segments import LineSegment, create_segment, line_num_points
from A import SegmentType, StartMode, StarType, length_range, start_x_range, start_y_range, relative_location_range, \
    thickness_range, direction_range, colour_options, curviness_range


# --- Fake Helper Functions for Testing render and curve_between_lines ---

def fake_bezier_curve_t(t_values, p0, p1, p2):
    # For testing, return a simple linear interpolation from p0 to p2.
    p0, p2 = np.array(p0), np.array(p2)
    return np.column_stack((p0[0] + (p2[0]-p0[0])*t_values, p0[1] + (p2[1]-p0[1])*t_values))

def fake_un_normalised_vector_direction(p0, p2):
    p0, p2 = np.array(p0), np.array(p2)
    return p2 - p0

def fake_point_in_array(array, threshold):
    # For testing, always return index 0.
    return 0

# --- Tests for __init__ ---
@pytest.mark.parametrize("start_mode, provided_start_location, provided_split_point, expected_start_location, expected_split_point", [
    (StartMode.CONNECT_MID, 0.75, 0.3, 0.75, 0),
    (StartMode.SPLIT, 0.4, 0.25, 1, 0.25),
    (StartMode.CONNECT, 0.2, 0.9, 1, 0),
    (StartMode.JUMP, 0.8, 0.1, 1, 0),
])
def test_line_segment_init(start_mode, provided_start_location, provided_split_point,
                             expected_start_location, expected_split_point):
    seg = LineSegment(
        segment_type=SegmentType.LINE,
        start=(0, 0),
        start_mode=start_mode,
        length=10,
        relative_angle=45,
        start_thickness=2,
        end_thickness=1.5,
        colour="red",
        curviness=0.3,
        curve_left=True,  # this value is used if curve_90 is not provided
        curve_location=0.7,
        start_location=provided_start_location,
        split_point=provided_split_point
    )
    assert seg.start_location == expected_start_location, f"Expected start_location {expected_start_location}, got {seg.start_location}"
    assert seg.split_point == expected_split_point, f"Expected split_point {expected_split_point}, got {seg.split_point}"

# --- Test for calculate_end ---
@pytest.mark.parametrize("start, length, relative_angle, start_mode, prev_angle, expected_absolute_angle, expected_end", [
    # For JUMP mode, absolute_angle = relative_angle.
    ((0, 0), 10, 90, StartMode.JUMP, 30, 90, (10 * math.cos(math.radians(90)), 10 * math.sin(math.radians(90)))),
    # For CONNECT (or other) mode, absolute_angle = (prev_angle + relative_angle) mod 360.
    ((1, 1), 5, 45, StartMode.CONNECT, 15, (15+45)%360, (1 + 5*math.cos(math.radians(60)), 1 + 5*math.sin(math.radians(60)))),
])
def test_calculate_end(start, length, relative_angle, start_mode, prev_angle, expected_absolute_angle, expected_end):
    seg = LineSegment(
        segment_type=SegmentType.LINE,
        start=start,
        start_mode=start_mode,
        length=length,
        relative_angle=relative_angle,
        start_thickness=2,
        end_thickness=1,
        colour="blue",
        curviness=0,
        curve_left=True,
        curve_location=0.5,
        start_location=1,
        split_point=0
    )
    seg.calculate_end(prev_angle)
    assert math.isclose(seg.absolute_angle, expected_absolute_angle, rel_tol=1e-5), f"Expected absolute_angle {expected_absolute_angle}, got {seg.absolute_angle}"
    expected_end = (round(expected_end[0], 5), round(expected_end[1], 5))
    actual_end = (round(seg.end[0], 5), round(seg.end[1], 5))
    assert actual_end == expected_end, f"Expected end {expected_end}, got {actual_end}"

# --- Test for curve_between_lines ---
"""
@pytest.mark.parametrize("p0, p1, p2, p3, p4, colour, connecting_array", [
    (
        (0,0), (1,2), (2,2), (3,2), (4,0), "red",
        np.array([[0,0], [0.5, 0.5]])
    )
])

def test_curve_between_lines(monkeypatch, p0, p1, p2, p3, p4, colour, connecting_array):
    import Segments
    monkeypatch.setattr(Segments, "bezier_curve_t", fake_bezier_curve_t)
    # We don't want actual plotting, so patch plt.fill.
    monkeypatch.setattr(plt, "fill", lambda x, y, color, alpha: None)
    seg = LineSegment(
        segment_type=SegmentType.LINE,
        start=(0,0),
        start_mode=StartMode.CONNECT,
        length=10,
        relative_angle=45,
        start_thickness=2,
        end_thickness=1,
        colour="red",
        curviness=0,
        curve_left=True,
        curve_location=0.5,
        start_location=1,
        split_point=0
    )
    # Set a known points_array for later concatenation.
    seg.points_array = np.array([[10,10], [20,20]])
    result = seg.curve_between_lines(p0, p1, p2, p3, p4, colour, connecting_array)
    left_curve = fake_bezier_curve_t(np.linspace(0,1,20), p0, p1, p2)
    right_curve = fake_bezier_curve_t(np.linspace(0,1,20), p2, p3, p4)
    expected = np.concatenate((left_curve, right_curve, seg.points_array), axis=0)
    assert np.allclose(result, expected), "curve_between_lines did not return the expected concatenated array."
"""
@pytest.mark.parametrize("p0, p1, p2, p3, p4, colour", [
    ((0, 0), (1, 2), (2, 2), (3, 2), (4, 0), "red")
])
def test_curve_between_lines(monkeypatch, p0, p1, p2, p3, p4, colour):
    import Segments
    # Patch bezier_curve_t in Segments.
    monkeypatch.setattr(Segments, "bezier_curve_t", fake_bezier_curve_t)

    # Capture plt.fill arguments.
    captured = {}

    def fake_fill(x, y, color, alpha):
        captured["boundary_x"] = x
        captured["boundary_y"] = y
        captured["color"] = color
        captured["alpha"] = alpha

    monkeypatch.setattr(plt, "fill", fake_fill)

    # Create a LineSegment and set its points_array to a known fixed array.
    seg = LineSegment(
        segment_type=SegmentType.LINE,
        start=(0, 0),
        start_mode=StartMode.CONNECT,
        length=10,
        relative_angle=45,
        start_thickness=2,
        end_thickness=1,
        colour="red",
        curviness=0,  # so no bezier curve in render
        curve_left=True,
        curve_location=0.5,
        start_location=1,
        split_point=0
    )
    # Set points_array to a fixed array with 5 points.
    seg.points_array = np.array([
        [10, 10],
        [20, 20],
        [30, 30],
        [40, 40],
        [50, 50]
    ])

    curve_start_point = 1
    curve_end_point = 3
    connecting_array = seg.points_array[curve_start_point: curve_end_point ]  # points_array[1:3]

    # Call curve_between_lines.
    result = seg.curve_between_lines(p0, p1, p2, p3, p4, colour, connecting_array)

    # Check that plt.fill was called with a boundary computed as:
    # boundary = np.concatenate([expected_connecting_array[::-1], left_curve, right_curve], axis=0)
    t_values = np.linspace(0, 1, 20)
    left_curve = fake_bezier_curve_t(t_values, p0, p1, p2)
    right_curve = fake_bezier_curve_t(t_values, p2, p3, p4)
    expected_boundary = np.concatenate([connecting_array[::-1], left_curve, right_curve], axis=0)

    # Reconstruct the boundary from captured x and y.
    captured_boundary = np.column_stack((captured["boundary_x"], captured["boundary_y"]))
    assert np.allclose(captured_boundary, expected_boundary), "plt.fill was called with an incorrect boundary."

    # The function returns np.concatenate((left_curve, right_curve, self.points_array), axis=0)
    expected_return = np.concatenate((left_curve, right_curve, seg.points_array), axis=0)
    assert np.allclose(result, expected_return), "curve_between_lines did not return the expected concatenated array."


# --- Test for render ---
@pytest.mark.parametrize("start, length, relative_angle, prev_angle, prev_array", [
    ((0,0), 10, 0, 0, np.array([[0,0], [5,0], [10,0]])),
    ((1,1), 5, 90, 45, np.array([[1,1], [1,3], [1,5]]))
])
def test_render(monkeypatch, start, length, relative_angle, prev_angle, prev_array):
    import Segments
    monkeypatch.setattr(Segments, "point_in_array", fake_point_in_array)
    monkeypatch.setattr(Segments, "un_normalised_vector_direction", fake_un_normalised_vector_direction)
    seg = LineSegment(
        segment_type=SegmentType.LINE,
        start=start,
        start_mode=StartMode.CONNECT,
        length=length,
        relative_angle=relative_angle,
        start_thickness=2,
        end_thickness=1,
        colour="green",
        curviness=0,  # so linear interpolation is used
        curve_left=True,
        curve_location=0.5,
        start_location=1,
        split_point=0
    )
    seg.calculate_end(prev_angle)
    seg.render(prev_array, prev_angle, "green", np.array([2,2]), scale=1, ax_n=None)
    expected_points = np.column_stack((
        np.linspace(seg.start[0], seg.end[0], line_num_points),
        np.linspace(seg.start[1], seg.end[1], line_num_points)
    ))
    expected_points = np.round(expected_points, 3)
    assert np.allclose(seg.points_array, expected_points), "render did not create the expected points_array for a straight line."

# --- Test for mutate ---
# We assume that mutate calls instance methods mutate_choice, mutate_val, and mutate_colour.
# We patch these methods on the instance so that mutation does nothing.

@pytest.mark.parametrize("mutation_rate", [0, 0.5, 1])
def test_mutate(monkeypatch, mutation_rate):
    # Create a LineSegment with valid attributes; note length is set to 60 (within length_range, assumed (0,60)).
    seg = LineSegment(
        segment_type=SegmentType.LINE,
        start=(50,100),
        start_mode=StartMode.JUMP,
        length=60,
        relative_angle=45,
        start_thickness=2,
        end_thickness=1,
        colour="red",
        curviness=0.7,
        curve_left=True,
        curve_location=0.4,
        start_location=1,
        split_point=0
    )
    seg.mutate_choice = lambda val, options, rate: val
    seg.mutate_val = lambda val, value_range, rate: val
    seg.mutate_colour = lambda col, rate: col
    orig_values = {
        "start_mode": seg.start_mode,
        "start_thickness": seg.start_thickness,
        "end_thickness": seg.end_thickness,
        "relative_angle": seg.relative_angle,
        "colour": seg.colour,
        "length": seg.length,
        "curviness": seg.curviness,
        "curve_left": seg.curve_left,
        "curve_location": seg.curve_location,
        "start_location": seg.start_location,
        "split_point": seg.split_point,
    }
    seg.mutate(mutation_rate)
    for attr, original in orig_values.items():
        mutated = getattr(seg, attr)
        assert mutated == original, f"Attribute {attr} changed after mutate: expected {original}, got {mutated}"

# --- Test for mutate ---
# We assume that the mutate method calls self.mutate_choice, self.mutate_val, and self.mutate_colour.
# For testing, we monkeypatch these methods on the instance to simply return the input value unchanged.
@pytest.mark.parametrize("mutation_rate", [0, 0.5, 1])
def test_mutate(monkeypatch, mutation_rate):
    # Create a LineSegment with fixed attributes.
    seg = LineSegment(
        segment_type=SegmentType.LINE,
        start=(0,0),
        start_mode=StartMode.CONNECT,
        length=10,
        relative_angle=45,
        start_thickness=2,
        end_thickness=1.5,
        colour="red",
        curviness=0.5,
        curve_left=False,
        curve_location=0.7,
        start_location=1,
        split_point=0
    )
    # Monkeypatch the mutation helper methods on the instance.
    seg.mutate_choice = lambda val, options, rate: val
    seg.mutate_val = lambda val, value_range, rate: val
    seg.mutate_colour = lambda col, rate: col

    # Store original attribute values.
    orig_values = {
        "start_mode": seg.start_mode,
        "start_thickness": seg.start_thickness,
        "end_thickness": seg.end_thickness,
        "relative_angle": seg.relative_angle,
        "colour": seg.colour,
        "length": seg.length,
        "curviness": seg.curviness,
        "curve_left": seg.curve_left,
        "curve_location": seg.curve_location,
        "start_location": seg.start_location,
        "split_point": seg.split_point,
    }
    # Call mutate.
    seg.mutate(mutation_rate)
    # With our patched mutation helpers, none of the values should change.
    for attr, original in orig_values.items():
        mutated = getattr(seg, attr)
        assert mutated == original, f"Attribute {attr} changed after mutate: expected {original}, got {mutated}"


def test_line_mutation_attributes_within_range():
    # Create a LineSegment with valid attributes
    line = LineSegment(SegmentType.LINE, (50,100), StartMode.JUMP, 60, 0, 2, 1, 'red', 0.7, True, 0.4, 0, 0)
    # Run mutation several times and check that mutated attributes remain within defined ranges.
    for count in range(10):
        line.mutate()
        assert line.segment_type == SegmentType.LINE
        assert isinstance(line.start_mode, StartMode)
        assert start_x_range[0] <= line.start[0] <= start_x_range[1]
        assert start_y_range[0] <= line.start[1] <= start_y_range[1]
        assert relative_location_range[0] <= line.start_location <= relative_location_range[1]
        assert relative_location_range[0] <= line.split_point <= relative_location_range[1]
        assert thickness_range[0] <= line.end_thickness <= thickness_range[1]
        assert direction_range[0] <= line.relative_angle <= direction_range[1]
        assert line.colour in colour_options
        assert length_range[0] <= line.length <= length_range[1]
        assert thickness_range[0] <= line.start_thickness <= thickness_range[1]
        assert curviness_range[0] <= line.curviness <= curviness_range[1]
        #assert isinstance(line.curve_left, bool)
        assert line.curve_left in [True, False], f"curve_left {line.curve_left} is not boolean."
        assert relative_location_range[0] <= line.curve_location <= relative_location_range[1]

    print("LineSegment tests passed successfully!")

