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
    "prev_array, prev_angle, start_point, start_mode, length, relative_angle, start_thickness, end_thickness, colour, curve_location, curve_left, curviness, split_point, start_location",
    [
        # First case: basic jump with no curviness.
        (np.array([[i, 0] for i in range(0, 100)]), 0, (50,100), StartMode.JUMP, 70, 0, 1, 2, 'red', 0, True, 0.4, 0, 0),
        # Second case: different length and angle, different colour and curvature.
        (np.array([[0, i] for i in range(0, 100)]), 90, (10,10), StartMode.JUMP, 100, 45, 3, 2, 'green', 0.5, True, 0, 0, 0),
        # Third case: CONNECT type.
        (np.array([[0, i] for i in range(0, 100)]), 90, (10,10), StartMode.CONNECT, 100, 45, 3, 2, 'green', 0.5, True, 0, 0, 0),
        # Forth case: CONNECT_MID type.
        (np.array([[0, i] for i in range(0, 100)]), 90, (10,10), StartMode.CONNECT_MID, 100, 45, 3, 2, 'green', 0.5, True, 0, 0, 0),
        # Fifth case: SPLIT type.
        (np.array([[0, i] for i in range(0, 100)]), 90, (10,10), StartMode.SPLIT, 100, 45, 3, 2, 'green', 0.5, True, 0, 0, 0),
    ]
)
def test_render_integration(monkeypatch, prev_array, prev_angle, start_point, start_mode, length, relative_angle, start_thickness, end_thickness, colour,
                            curve_location, curve_left, curviness, split_point, start_location):
    import Segments
    monkeypatch.setattr(Segments, "line_num_points", 10)
    # Set up dummy previous parameters.
    prev_colour = 'blue'
    prev_thickness_array = np.linspace(1, 101, 100)

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
        curve_location,  # curve_location from parameter
        curve_left,  # curve_left flag from parameter
        curviness,  # curviness from parameter
        split_point,  # split point from parameter
        start_location  # start location for CONNECT_MID if needed
    )

    # Execute the render method.
    line.render(prev_array, prev_angle, prev_colour, prev_thickness_array, scale=1, ax_n=dummy_ax)

    # Validate that points_array exists and is correctly shaped.
    assert hasattr(line, 'points_array'), "points_array was not set during rendering."
    assert line.points_array.shape == (10 , 2), "points_array does not have the expected shape."

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