from A import SegmentType, StartMode, line_num_points
from Segments import LineSegment
import pytest
import numpy as np

# Dummy axes object to capture plot calls.
class DummyAx:
    def __init__(self):
        self.plot_calls = []  # Collects calls to plot.

    def plot(self, x, y, **kwargs):
        self.plot_calls.append((x, y, kwargs))

@pytest.mark.parametrize(
    "prev_array, prev_angle, start_point, start_mode, length, relative_angle, start_thickness, end_thickness, colour, curve_location, curve_left, curviness, split_point, start_location, expected_points_array",
    [
        # First case: basic jump with no curviness.
        (np.array([[i, 0] for i in range(0, 10)]), 0, (50,100), StartMode.JUMP, 70, 0, 1, 2, 'red', 0, True, 0, 0, 0,np.array([
    [50.0, 100.0],
    [57.778, 100.0],
    [65.556, 100.0],
    [73.333, 100.0],
    [81.111, 100.0],
    [88.889, 100.0],
    [96.667, 100.0],
    [104.444, 100.0],
    [112.222, 100.0],
    [120.0, 100.0] ]) ),
        # Second case: different length and angle, different colour and curvature.
        (np.array([[0, i] for i in range(0, 10)]), 90, (10,10), StartMode.JUMP, 50, 45, 3, 2, 'green', 0.5, True, 0.4, 0, 0, np.array([
    [10.0, 10.0],
    [11.135, 16.722],
    [12.968, 22.745],
    [15.5, 28.071],
    [18.73, 32.697],
    [22.658, 36.626],
    [27.285, 39.856],
    [32.61, 42.387],
    [38.633, 44.22],
    [45.355, 45.355]
]) ),
        # Third case: CONNECT type.
        (np.array([[0, i] for i in range(0, 10)]), 90, (10,10), StartMode.CONNECT, 100, 45, 3, 2, 'green', 0, True, 0, 0, 0, np.array([
    [0.0, 9.0],
    [-7.857, 16.857],
    [-15.713, 24.713],
    [-23.57, 32.57],
    [-31.427, 40.427],
    [-39.284, 48.284],
    [-47.14, 56.14],
    [-54.997, 63.997],
    [-62.854, 71.854],
    [-70.711, 79.711]
]) ),
        # Forth case: CONNECT_MID type.
        (np.array([[0, i] for i in range(0, 10)]), 30, (10,10), StartMode.CONNECT_MID, 60, 45, 3, 2, 'green', 0, True, 0, 0, 0.1, np.array([
    [0.0, 1.0],
    [1.725, 7.44],
    [3.451, 13.879],
    [5.176, 20.319],
    [6.902, 26.758],
    [8.627, 33.198],
    [10.353, 39.637],
    [12.078, 46.077],
    [13.804, 52.516],
    [15.529, 58.956]
]) ),
        # Fifth case: SPLIT type.
        (np.array([[0, i] for i in range(0, 10)]), 90, (10,10), StartMode.SPLIT, 30, 100, 3, 2, 'green', 0, True, 0, 0.6, 0, np.array([
    [16.413, 11.894],
    [13.13, 11.315],
    [9.848, 10.736],
    [6.565, 10.158],
    [3.282, 9.579],
    [0.0, 9.0],
    [-3.283, 8.421],
    [-6.566, 7.842],
    [-9.849, 7.263],
    [-13.131, 6.685]
]) ),
    ]
)
def test_render_integration(monkeypatch, prev_array, prev_angle, start_point, start_mode, length, relative_angle, start_thickness, end_thickness, colour,
                            curve_location, curve_left, curviness, split_point, start_location, expected_points_array):
    import Segments
    import numpy as np
    monkeypatch.setattr(Segments, "line_num_points", 10)
    # Set up dummy previous parameters.
    prev_colour = 'blue'
    prev_thickness_array = np.linspace(1, 11, 10)

    dummy_ax = DummyAx()

    # Create the LineSegment instance with parameterized values.
    line = LineSegment(
        SegmentType.LINE,  # segment type
        start_point,  # start point
        start_mode,  # start mode from parameter
        length,  # length from parameter
        relative_angle,  # relative angle from parameter
        start_thickness,  # start thickness from parameter
        end_thickness,  # end thickness from parameter
        colour,  # colour from parameter
        curviness,  # curviness from parameter
        curve_left,  # curve_left flag from parameter
        curve_location,  # curve_location from parameter
        start_location,  # start location for CONNECT_MID if needed
        split_point # split point from parameter

    )

    # Execute the render method.
    line.render(prev_array, prev_angle, prev_colour, prev_thickness_array, scale=1, ax_n=dummy_ax)

    # Validate that points_array exists and is correctly shaped.
    assert hasattr(line, 'points_array'), "points_array was not set during rendering."
    assert line.points_array.shape == (10 , 2), "points_array does not have the expected shape."
    #np.testing.assert_allclose(line.points_array, expected_points_array, atol=1e-3,
    #                           err_msg="points_array does not match expected values.")
    import numpy as np

    # Assume line.points_array and expected_points_array are defined
    if not np.allclose(line.points_array, expected_points_array, atol=1e-3):
        print("Mismatch detected:")
        print("Expected points_array:")
        print(expected_points_array)
        print("\nComputed (real) points_array:")
        print(line.points_array)
        print("\nDifference (Real - Expected):")
        print(line.points_array - expected_points_array)

    # Now assert that the arrays are close (this will fail if they aren't)
    np.testing.assert_allclose(line.points_array, expected_points_array, atol=1e-3,
                               err_msg="points_array does not match expected values.")

    # Validate that thickness_array exists and has the expected length.
    assert hasattr(line, 'thickness_array'), "thickness_array was not set during rendering."
    assert line.thickness_array.shape == (10,), "thickness_array does not have the expected length."

    # Verify the number of plot calls equals (line_num_points - 1).
    expected_calls = 10 - 1
    assert len(
        dummy_ax.plot_calls) == expected_calls, f"Expected {expected_calls} plot calls but got {len(dummy_ax.plot_calls)}."

    # Check that the first plot call uses the first two computed points.
    x_values = line.points_array[:, 0]
    y_values = line.points_array[:, 1]
    first_call = dummy_ax.plot_calls[0]
    np.testing.assert_allclose(first_call[0], [x_values[0], x_values[1]], atol=1e-3,
                               err_msg="X coordinates in the first plot call do not match expected values.")
    np.testing.assert_allclose(first_call[1], [y_values[0], y_values[1]], atol=1e-3,
                               err_msg="Y coordinates in the first plot call do not match expected values.")