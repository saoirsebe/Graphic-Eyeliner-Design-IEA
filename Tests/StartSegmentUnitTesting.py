import numpy as np
import pytest
import math
import matplotlib.pyplot as plt

from Segments import StarSegment
from A import SegmentType, StartMode, StarType, radius_range, arm_length_range, num_points_range, asymmetry_range, start_x_range, start_y_range, thickness_range, direction_range, colour_options



# --- Fake helper functions for testing ---
def fake_point_in_array_factory(fixed_index):
    def fake_point_in_array(array, threshold):
        return fixed_index

    return fake_point_in_array


def fake_star_create(num_points, center, radius, arm_length, asymmetry, star_type, absolute_angle, fill):
    # For testing, return fixed arrays.
    star_points = np.array([[center[0] + 1, center[1] + 1],
                            [center[0] + 2, center[1] + 2],
                            [center[0] + 3, center[1] + 3]])
    star_arm_points = np.array([[center[0] - 1, center[1] - 1],
                                [center[0] - 0.5, center[1] - 0.5],
                                [center[0], center[1]]])
    return star_points, star_arm_points


def fake_un_normalised_vector_direction(p0, p2):
    return np.array(p2) - np.array(p0)


# Patch StarGeneration.create_star in our tests.
@pytest.fixture(autouse=True)
def patch_star_generation(monkeypatch):
    monkeypatch.setattr("Segments.StarGeneration.create_star", fake_star_create)


# --- Test _update_start_and_center ---
@pytest.mark.parametrize("mode, prev_array, fixed_index, expected_start", [
    # For CONNECT with short prev_array (length <= num_points_range[1], where num_points_range[1]=8)
    (StartMode.CONNECT, [(i, i) for i in range(7)], 3, (3, 3)),
    # For CONNECT with long prev_array (length > 8)
    (StartMode.CONNECT, [(i, i) for i in range(20)], None, (19, 19)),
])
def test_update_start_and_center(monkeypatch, mode, prev_array, fixed_index, expected_start):
    seg = StarSegment(SegmentType.STAR, (999, 999), "red", StarType.STRAIGHT,
                      radius=5, arm_length=4, num_points=5, asymmetry=0,
                      start_mode=mode, end_thickness=1, relative_angle=45, fill=False)
    if fixed_index is not None:
        monkeypatch.setattr("Segments.point_in_array", fake_point_in_array_factory(fixed_index))
    else:
        monkeypatch.setattr("Segments.point_in_array", fake_point_in_array_factory(0))
    seg._update_start_and_center(prev_array)
    assert seg.start == expected_start


# --- Test _compute_absolute_angle ---
@pytest.mark.parametrize("mode, prev_angle, relative_angle, expected_abs", [
    (StartMode.JUMP, 30, 45, 45),
    (StartMode.CONNECT, 30, 45, (30 + 45) % 360),
    (StartMode.CONNECT_MID, 100, 10, (100 + 10) % 360),
])
def test_compute_absolute_angle(mode, prev_angle, relative_angle, expected_abs):
    seg = StarSegment(SegmentType.STAR, (0, 0), "red", StarType.STRAIGHT,
                      radius=5, arm_length=4, num_points=5, asymmetry=0,
                      start_mode=mode, end_thickness=1, relative_angle=relative_angle, fill=False)
    seg._compute_absolute_angle(prev_angle)
    assert seg.absolute_angle == expected_abs


# --- Test _compute_star_geometry ---
def test_compute_star_geometry(monkeypatch):
    center = (10, 10)
    seg = StarSegment(SegmentType.STAR, (0, 0), "red", StarType.STRAIGHT,
                      radius=5, arm_length=4, num_points=5, asymmetry=0,
                      start_mode=StartMode.CONNECT, end_thickness=1, relative_angle=45, fill=False)
    seg.center = center
    # Expected star_points and star_arm_points come from fake_star_create.
    star_points, star_arm_points = fake_star_create(seg.num_points, center, seg.radius, seg.arm_length, seg.asymmetry,
                                                    seg.star_type, seg.absolute_angle, seg.fill)
    transformation_vector = (center[0] - star_arm_points[-1][0], center[1] - star_arm_points[-1][1])
    expected_star = np.array(
        [(pt[0] + transformation_vector[0], pt[1] + transformation_vector[1]) for pt in star_points])
    expected_arm = np.array(
        [(pt[0] + transformation_vector[0], pt[1] + transformation_vector[1]) for pt in star_arm_points])
    transformed_star, transformed_arm = seg._compute_star_geometry()
    assert np.allclose(transformed_star, expected_star)
    assert np.allclose(transformed_arm, expected_arm)


# --- Test render ---
@pytest.mark.parametrize("prev_array, prev_angle", [
    ([(i, i) for i in range(10)], 30),
    ([(i, i) for i in range(20)], 45)
])
def test_render(monkeypatch, prev_array, prev_angle):
    # Patch the helper methods to return known values.
    fixed_star_points = np.array([[1, 1], [2, 2], [3, 3]])
    fixed_arm_points = np.array([[0, 0], [0.5, 0.5], [1, 1]])
    monkeypatch.setattr(StarSegment, "_update_start_and_center", lambda self, pa: setattr(self, "start", (
        prev_array[-1] if len(prev_array) > 15 else prev_array[0])) or setattr(self, "center", self.start))
    monkeypatch.setattr(StarSegment, "_compute_absolute_angle",
                        lambda self, pa: setattr(self, "absolute_angle", (prev_angle + self.relative_angle) % 360))
    monkeypatch.setattr(StarSegment, "_compute_star_geometry", lambda self: (fixed_star_points, fixed_arm_points))

    seg = StarSegment(SegmentType.STAR, (0, 0), "red", StarType.STRAIGHT,
                      radius=5, arm_length=4, num_points=5, asymmetry=0,
                      start_mode=StartMode.CONNECT, end_thickness=1, relative_angle=45, fill=False)
    seg.render(prev_array, prev_angle, "blue", np.array(range(len(prev_array))), scale=1, ax_n=None)
    assert np.allclose(seg.points_array, fixed_star_points)
    assert np.allclose(seg.arm_points_array, fixed_arm_points)


# --- Test mutate ---
# This test forces mutation to occur by patching np.random.random
# (but does not alter np.random.normal or np.random.choice).
@pytest.mark.parametrize("attribute, initial_value, value_range", [
    ("radius", 5, radius_range),         # e.g. radius_range might be (1, 10)
    ("arm_length", 4, arm_length_range),   # e.g. arm_length_range might be (1, 10)
    ("num_points", 7, num_points_range),   # e.g. num_points_range might be (5, 10)
    ("asymmetry", 0, asymmetry_range),     # e.g. asymmetry_range might be (0, 5)
])
def test_star_segment_mutate_numerical(monkeypatch, attribute, initial_value, value_range):
    seg = StarSegment(
        SegmentType.STAR, (0,0), "red", StarType.STRAIGHT,
        radius=5, arm_length=4, num_points=7, asymmetry=0,
        start_mode=StartMode.CONNECT, end_thickness=2, relative_angle=45, fill=True
    )
    setattr(seg, attribute, initial_value)
    orig = getattr(seg, attribute)
    # Force mutation by making np.random.random always return 0
    monkeypatch.setattr(np.random, "random", lambda: 0.0)
    seg.mutate(1.0)  # mutation_rate = 1.0 forces mutation
    new_val = getattr(seg, attribute)
    # Check that the mutated value is within the expected range and is not equal to the original.
    assert value_range[0] <= new_val <= value_range[1], f"{attribute} out of range: {new_val} not in {value_range}"
    assert new_val != orig, f"Expected {attribute} to change from {orig}, but got {new_val}"