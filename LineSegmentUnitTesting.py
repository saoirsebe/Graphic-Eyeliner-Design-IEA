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
        curve_left=True,
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
    seg._curve_between_lines(p0, p1, p2, p3, p4, colour, connecting_array)

    # Check that plt.fill was called with a boundary computed as:
    t_values = np.linspace(0, 1, 20)
    left_curve = fake_bezier_curve_t(t_values, p0, p1, p2)
    right_curve = fake_bezier_curve_t(t_values, p2, p3, p4)
    expected_boundary = np.concatenate([connecting_array[::-1], left_curve, right_curve], axis=0)

    # Reconstruct the boundary from captured x and y.
    captured_boundary = np.column_stack((captured["boundary_x"], captured["boundary_y"]))
    assert np.allclose(captured_boundary, expected_boundary), "plt.fill was called with an incorrect boundary."


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

    seg.render(prev_array, prev_angle, "green", np.array([2,2]), scale=1, ax_n=None)
    expected_points = np.column_stack((
        np.linspace(seg.start[0], seg.end[0], line_num_points),
        np.linspace(seg.start[1], seg.end[1], line_num_points)
    ))
    expected_points = np.round(expected_points, 3)
    assert np.allclose(seg.points_array, expected_points), "render did not create the expected points_array for a straight line."

# --- Test for update_start_from_prev ---
import numpy as np
import pytest
from Segments import LineSegment
from A import StartMode


# We'll create a fake point_in_array that returns a fixed index if provided.
def fake_point_in_array_factory(fixed_index):
    def fake_point_in_array(array, threshold):
        return fixed_index

    return fake_point_in_array


@pytest.mark.parametrize("mode, prev_length, fixed_index, exp_mode_branch", [
    #num_points_range[1] = 8
    # Case A: CONNECT, length > 8: branch 1 should be taken.
    (StartMode.CONNECT, 20, None, "branch1"),
    # Case B: CONNECT, length <= 8: branch 2.
    (StartMode.CONNECT, 7, 3, "branch2"),
    # Case C: SPLIT, length > 8: branch 1.
    (StartMode.SPLIT, 20, None, "branch1"),
    # Case D: SPLIT, length <= 8: branch 2.
    (StartMode.SPLIT, 7, 3, "branch2"),
    # Case E: CONNECT_MID, length > 0.
    (StartMode.CONNECT_MID, 10, 2, "connect_mid"),
])
def test_update_start_from_prev(monkeypatch, mode, prev_length, fixed_index, exp_mode_branch):
    # Create a previous array as a list of tuples: e.g., [(0,0), (1,1), ..., (n-1, n-1)]
    prev_array = [(i, i) for i in range(prev_length)]
    # Create a thickness array as a numpy array of 0,1,2,...,prev_length-1
    prev_thickness_array = np.array(list(range(prev_length)))

    # Create a dummy LineSegment. (Other attributes are not relevant for this test.)
    seg = LineSegment(
        segment_type=None,  # not used here
        start=(999, 999),  # dummy initial value
        start_mode=mode,
        length=10,
        relative_angle=45,
        start_thickness=0,
        end_thickness=1,
        colour="black",
        curviness=0,
        curve_left=True,
        curve_location=0.5,
        start_location=0.5,
        split_point=0.3
    )

    # For branches where fixed_index is needed, patch point_in_array accordingly.
    # Otherwise, we don't care (it won't be used).
    if fixed_index is not None:
        monkeypatch.setattr("Segments.point_in_array", fake_point_in_array_factory(fixed_index))
    else:
        # Provide a default fake that returns 0.
        monkeypatch.setattr("Segments.point_in_array", fake_point_in_array_factory(0))

    # Call update_start_from_prev
    ret = seg.update_start_from_prev(prev_array, prev_thickness_array, prev_length)

    # Now, check based on the expected branch.
    if exp_mode_branch == "branch1":
        # For CONNECT or SPLIT with length > 15:
        expected_start = prev_array[-1]
        expected_thickness = prev_thickness_array[-1]
        assert seg.start == expected_start, f"Expected start {expected_start}, got {seg.start}"
        assert seg.start_thickness == expected_thickness, f"Expected start_thickness {expected_thickness}, got {seg.start_thickness}"
        assert ret is None, f"Expected return None, got {ret}"
    elif exp_mode_branch == "branch2":
        # For CONNECT or SPLIT with length <= 15:
        # The fake point_in_array returns fixed_index.
        expected_start = prev_array[fixed_index]
        assert seg.start == expected_start, f"Expected start {expected_start}, got {seg.start}"
        # start_thickness is not updated in this branch.
        assert ret is None, f"Expected return None, got {ret}"
    elif exp_mode_branch == "connect_mid":
        # For CONNECT_MID with len(prev_array) > 0:
        # We expect point_in_array to be called with self.start_location and return fixed_index.
        expected_start = prev_array[fixed_index]
        assert seg.start == expected_start, f"Expected start {expected_start}, got {seg.start}"
        assert ret == fixed_index, f"Expected return {fixed_index}, got {ret}"
    else:
        pytest.fail("Unexpected branch.")


@pytest.mark.parametrize("prev_length", [0])
def test_update_start_from_prev_empty_array(monkeypatch, prev_length):
    # When prev_array is empty, the method should raise a ValueError.
    prev_array = []
    prev_thickness_array = np.array([])
    seg = LineSegment(
        segment_type=None,
        start=(999, 999),
        start_mode=StartMode.CONNECT,
        length=10,
        relative_angle=45,
        start_thickness=0,
        end_thickness=1,
        colour="black",
        curviness=0,
        curve_left=True,
        curve_location=0.5,
        start_location=0.5,
        split_point=0.3
    )
    with pytest.raises(ValueError):
        seg.update_start_from_prev(prev_array, prev_thickness_array, prev_length)


# --- Test for compute_points_array_straight ---
def test_compute_points_array_straight():
    # Create a segment with a known start and end.
    seg = LineSegment(
        segment_type=SegmentType.LINE,
        start=(0, 0),
        start_mode=StartMode.CONNECT,
        length=10,
        relative_angle=0,
        start_thickness=2,
        end_thickness=1,
        colour="blue",
        curviness=0,
        curve_left=True,
        curve_location=0.5,
        start_location=1,
        split_point=0
    )
    # Manually set end.
    seg.end = (10, 0)
    x_vals, y_vals = seg.compute_points_array_straight()
    expected_x = np.linspace(0, 10, line_num_points)
    expected_y = np.linspace(0, 0, line_num_points)
    expected = np.column_stack((expected_x, expected_y))
    expected = np.round(expected, 3)
    assert np.allclose(seg.points_array,
                       expected), "compute_points_array_straight did not produce expected points_array."
    # Also, verify that x_vals and y_vals match.
    assert np.allclose(x_vals, expected_x)
    assert np.allclose(y_vals, expected_y)


# --- Test for compute_points_array_curved ---
def test_compute_points_array_curved(monkeypatch):
    import Segments
    # Patch un_normalised_vector_direction and bezier_curve_t.
    monkeypatch.setattr(Segments, "un_normalised_vector_direction", lambda p0, p2: np.array(p2) - np.array(p0))
    monkeypatch.setattr(Segments, "bezier_curve_t", fake_bezier_curve_t)
    seg = LineSegment(
        segment_type=SegmentType.LINE,
        start=(0, 0),
        start_mode=StartMode.CONNECT,
        length=10,
        relative_angle=0,
        start_thickness=2,
        end_thickness=1,
        colour="blue",
        curviness=0.5,  # curved
        curve_left=True,
        curve_location=0.5,
        start_location=1,
        split_point=0
    )
    # Set an end point.
    seg.end = (10, 0)
    x_vals, y_vals = seg.compute_points_array_curved()
    # Our fake_bezier_curve_t does linear interpolation between p0 and p2.
    t = np.linspace(0, 1, line_num_points)
    expected = np.column_stack(((0 + (10 - 0) * t), (0 + (0 - 0) * t)))
    assert np.allclose(seg.points_array, expected), "compute_points_array_curved did not produce expected points_array."
    # Also check returned x_vals, y_vals.
    assert np.allclose(x_vals, expected[:, 0])
    assert np.allclose(y_vals, expected[:, 1])


# --- Test for apply_split_adjustments ---
@pytest.mark.parametrize("ax_n, prev_array, prev_angle", [
    # Case 1: ax_n is None -> blending branch skipped.
    (None, np.array([[0, 0], [1, 1], [2, 2], [3, 3]]), 0),
    # Case 2: ax_n is provided and prev_array length > 5.
    (object(), np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]]), 10)
])
def test_apply_split_adjustments(monkeypatch, ax_n, prev_array, prev_angle):
    # For testing, patch point_in_array to always return a fixed index, e.g. 1.
    monkeypatch.setattr("Segments.point_in_array", fake_point_in_array_factory(1))
    # Patch un_normalised_vector_direction to return a fixed vector, e.g., (5,5).
    monkeypatch.setattr("Segments.un_normalised_vector_direction", lambda p0, p2: np.array([5, 5]))
    # For simplicity, also patch curve_between_lines to do nothing.
    called_args = {}

    def fake_curve_between_lines(self, p0, p1, p2, p3, p4, prev_colour, connecting_array):
        called_args.update({"p0": p0, "p1": p1, "p2": p2, "p3": p3, "p4": p4, "prev_colour": prev_colour,
                            "connecting_array": connecting_array})

    monkeypatch.setattr("Segments.LineSegment.curve_between_lines", fake_curve_between_lines)

    # Create a dummy segment with a known points_array and start.
    seg = LineSegment(
        segment_type=SegmentType.LINE,
        start=(0, 0),
        start_mode=StartMode.SPLIT,
        length=10,
        relative_angle=45,
        start_thickness=2,
        end_thickness=1,
        colour="red",
        curviness=0,  # use straight line for simplicity.
        curve_left=True,
        curve_location=0.5,
        start_location=1,
        split_point=0.3
    )
    # Set a fixed points_array.
    seg.points_array = np.array([[10, 10], [20, 20], [30, 30], [40, 40]])

    # Before any blend adjustments, apply the transformation:
    # transformation_vector = un_normalised_vector_direction(self.points_array[fixed_index], self.start)
    # With fixed_index=1, self.points_array[1] = [20,20], self.start = (0,0), so transformation_vector = [20,20]?
    # However, our patched un_normalised_vector_direction returns (5,5) regardless.
    # So new points_array = original points_array + (5,5):
    expected_points = seg.points_array + np.array([5, 5])
    expected_x = expected_points[:, 0]
    expected_y = expected_points[:, 1]

    # For this test, len_prev_array is provided as parameter.
    x_vals, y_vals = seg.apply_split_adjustments("blue", prev_array, prev_angle, ax_n, len(prev_array))

    # If ax_n is None, the blend branch is not executed.
    assert np.allclose(x_vals, expected_x), "apply_split_adjustments (no blend): x values not as expected."
    assert np.allclose(y_vals, expected_y), "apply_split_adjustments (no blend): y values not as expected."
    if ax_n:
        # When ax_n is provided, the function calls curve_between_lines.
        # Our fake_curve_between_lines simply captures its arguments.
        # Additionally, we can check that curve_between_lines was called by verifying our captured arguments.
        assert "p0" in called_args, "curve_between_lines was not called in blend branch."


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

