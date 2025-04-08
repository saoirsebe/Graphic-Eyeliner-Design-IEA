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
    "prev_array, prev_angle, start_point, start_mode, length, relative_angle, start_thickness, end_thickness, colour, curve_location, curve_left, curviness, split_location, start_location, expected_points_array",
    [
        # Case 1: basic jump with no curviness.
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
        # Case 2: different length and angle, different colour and curvature.
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
        # Case 3: CONNECT type.
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
        # Case 4: CONNECT_MID type.
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
        # Case 5: SPLIT type.
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
                            curve_location, curve_left, curviness, split_location, start_location, expected_points_array):
    import Segments
    import numpy as np
    monkeypatch.setattr(Segments, "line_num_points", 10)
    # Set up dummy previous parameters.
    prev_colour = 'blue'
    prev_thickness_array = np.linspace(1, 11, 10)

    dummy_ax = DummyAx()

    # Create the LineSegment instance with parameterised values.
    line = LineSegment(
        SegmentType.LINE,
        start_point,
        start_mode,
        length,
        relative_angle,
        start_thickness,
        end_thickness,
        colour,
        curviness,
        curve_left,
        curve_location,  # curve_location for if curviness > 0
        start_location,  # start location for CONNECT_MID
        split_location # split point for SPLIT

    )

    # Execute the render method.
    line.render(prev_array, prev_angle, prev_colour, prev_thickness_array, scale=1, ax_n=dummy_ax)

    # Validate that points_array exists and is correctly shaped.
    assert hasattr(line, 'points_array'), "points_array was not set during rendering."
    assert line.points_array.shape == (10 , 2), "points_array does not have the expected shape."
    # Assert that the computed points_array is close to the expected points.
    np.testing.assert_allclose(line.points_array, expected_points_array, atol=1e-3,
                               err_msg="points_array does not match expected values.")

    # Validate that thickness_array exists and has the expected values.
    if start_mode == StartMode.CONNECT:
        start_thickness = prev_thickness_array[-1]
    expected_thickness_array = np.linspace(start_thickness, end_thickness, 10)
    assert hasattr(line, 'thickness_array'), "thickness_array was not set during rendering."
    np.testing.assert_allclose(line.thickness_array, expected_thickness_array, atol=1e-3,
                               err_msg="thickness_array does not match expected values.")

    # Check that each plot call uses the computed points_array.
    for i, (x_vals, y_vals, _) in enumerate(dummy_ax.plot_calls):
        np.testing.assert_allclose(x_vals, line.points_array[i:i + 2, 0], atol=1e-3,
                                   err_msg=f"X coordinates in plot call {i} do not match expected values.")
        np.testing.assert_allclose(y_vals, line.points_array[i:i + 2, 1], atol=1e-3,
                                   err_msg=f"Y coordinates in plot call {i} do not match expected values.")