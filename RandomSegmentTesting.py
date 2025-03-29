import numpy as np
import pytest
import random
from Segments import random_segment
from A import SegmentType, StartMode, StarType, direction_range  # Import necessary enums from module A

# --- Fake Helper Functions for Testing ---

def fake_generate_valid_start():
    # Always return a fixed start.
    return (0, 0)

def fake_random_segment_colour(prev_colour):
    # Always return a fixed colour.
    return "blue"

def fake_random_normal_within_range(mean, stddev, value_range):
    # Remove randomness: simply return the mean.
    return mean

def fake_random_from_two_distributions(a, b, c, d, value_range):
    # For testing, always return the first distribution's mean.
    return a

def fake_create_segment(**kwargs):
    # Create a dummy segment object.
    class DummySegment:
        pass
    seg = DummySegment()
    for key, value in kwargs.items():
        setattr(seg, key, value)
    # Add a dummy render method so that if the production code calls render, it won't fail.
    seg.render = lambda *args, **kwargs: None
    return seg

def fake_are_points_collinear(corners):
    # Assume the points are never collinear.
    return False

def fake_generate_radius_arm_lengths(new_star_type):
    # Return fixed values for radius and arm_length.
    return (2, 4)

def fake_fix_overlaps_shape_overlaps(segment, lines_list):
    # Assume no overlaps occur.
    return 0

# --- Test Code Using Parameterization and Monkeypatch ---

# We will force random.random() to yield a specific value so that a particular branch is taken:
#   - forced value 0.5  => r < 0.7       => should produce a LINE segment.
#   - forced value 0.9  => 0.7 <= r < 0.95  => should produce a STAR segment.
#   - forced value 0.97 => r >= 0.95      => should produce an IRREGULAR_POLYGON segment.
@pytest.mark.parametrize("forced_random, expected_segment_type", [
    (0.5, SegmentType.LINE),
    (0.9, SegmentType.STAR),
    (0.97, SegmentType.IRREGULAR_POLYGON),
])
def test_random_segment(forced_random, expected_segment_type, monkeypatch):
    # Import the Segment module so we can patch its helper functions.
    import Segments

    # Patch helper functions in the Segment module with our fakes.
    monkeypatch.setattr(Segments, "generate_valid_start", fake_generate_valid_start)
    monkeypatch.setattr(Segments, "random_segment_colour", fake_random_segment_colour)
    monkeypatch.setattr(Segments, "random_normal_within_range", fake_random_normal_within_range)
    monkeypatch.setattr(Segments, "random_from_two_distributions", fake_random_from_two_distributions)
    monkeypatch.setattr(Segments, "create_segment", fake_create_segment)
    monkeypatch.setattr(Segments, "are_points_collinear", fake_are_points_collinear)
    monkeypatch.setattr(Segments, "generate_radius_arm_lengths", fake_generate_radius_arm_lengths)
    monkeypatch.setattr(Segments, "fix_overlaps_shape_overlaps", fake_fix_overlaps_shape_overlaps)

    # Also, patch random.choice and random.uniform to return deterministic values.
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(random, "uniform", lambda a, b: a)
    # Patch random.random() so that its first call returns our forced value.
    monkeypatch.setattr(random, "random", lambda: forced_random)

    # Call the function under test.
    segment = random_segment()

    # Check common attributes.
    assert hasattr(segment, "segment_type"), "Segment should have a 'segment_type' attribute."
    assert segment.segment_type == expected_segment_type, (
        f"Expected segment_type '{expected_segment_type}', got '{segment.segment_type}'"
    )
    # Check that the starting point and colour are set from our fake functions.
    assert segment.start == fake_generate_valid_start()
    assert segment.colour == "blue"

    # Now check branch-specific attributes.
    if expected_segment_type == SegmentType.LINE:
        # In the LINE branch, our forced random value (0.5) leads to:
        #   - new_segment_type = LINE.
        #   - random_start_mode = random.choice([...]) → first element → StartMode.CONNECT,
        #     since random.random() (0.5) is < 0.9.
        #   - Since segment_start equals eye_corner_start (both are (0,0) by our fakes),
        #     random_relative_angle = random_normal_within_range(22.5, 20, direction_range) → 22.5.
        #   - length = random_normal_within_range(30,20,length_range) → 30.
        #   - start_thickness = 1, end_thickness = 1.5.
        #   - curviness: since another random.random() call returns 0.5 (and 0.5 >= 0.3),
        #     it uses random_curviness() → random_normal_within_range(0.3,0.15,curviness_range) → 0.3.
        #   - curve_left = random.choice([True, False]) → True.
        #   - curve_location, start_location, split_point all equal 0.5.
        expected_line_attrs = {
            "start_mode": StartMode.CONNECT,
            "relative_angle": 135,
            "length": 30,
            "start_thickness": 1,
            "end_thickness": 1.5,
            "curviness": 0.3,
            "curve_left": True,
            "curve_location": 0.5,
            "start_location": 0.5,
            "split_point": 0.5,
        }
        for attr, expected_value in expected_line_attrs.items():
            actual = getattr(segment, attr, None)
            assert actual == expected_value, f"LINE: Expected {attr} to be {expected_value}, got {actual}"

    elif expected_segment_type == SegmentType.STAR:
        # In the STAR branch (forced_random = 0.9):
        #   - new_star_type = random.choice([...]) → first element → StarType.STRAIGHT.
        #   - generate_radius_arm_lengths returns (5,4).
        #   - start_mode = random.choice([StartMode.CONNECT, StartMode.JUMP]) → StartMode.CONNECT.
        #   - num_points = round(random_normal_within_range(5,2,num_points_range)) → round(5) = 5.
        #   - asymmetry: since random.random() returns 0.9 (>= 0.6), asymmetry = random_normal_within_range(2,2,asymmetry_range) → 2.
        #   - end_thickness = 2.
        #   - relative_angle = random.uniform(*direction_range) → direction_range[0].
        #   - fill = random.choice([True, False]) → True.
        expected_star_attrs = {
            "start_mode": StartMode.CONNECT,
            "radius": 2,
            "arm_length": 4,
            "num_points": 5,
            "asymmetry": 2,
            "star_type": StarType.STRAIGHT,
            "end_thickness": 2,
            "relative_angle": direction_range[0],
            "fill": True,
        }
        for attr, expected_value in expected_star_attrs.items():
            actual = getattr(segment, attr, None)
            assert actual == expected_value, f"STAR: Expected {attr} to be {expected_value}, got {actual}"

    elif expected_segment_type == SegmentType.IRREGULAR_POLYGON:
        # In the IRREGULAR_POLYGON branch (forced_random = 0.97):
        #   - start_mode is determined by: random.random() < 0.3 ? StartMode.CONNECT : StartMode.JUMP.
        #     Since forced_random (0.97) is not < 0.3, start_mode should be StartMode.JUMP.
        #   - end_thickness = random_normal_within_range(1,2,thickness_range) → 1.
        #   - relative_angle = random.uniform(*direction_range) → direction_range[0].
        #   - bounding_size_x = random_normal_within_range(30,20,random_shape_size_range) → 30.
        #     and bounding_size second element = random_normal_within_range(bounding_size_x,20,random_shape_size_range) → 30.
        #   - corners: n_of_corners = round(random_normal_within_range(5,1,num_points_range)) → round(5)=5.
        #     We then sort these corners; we can at least check that we have 5 corners.
        #   - lines_list should have the same length as n_of_corners (5).
        #   - fill = random.choice([True, False]) → True.
        expected_polygon_attrs = {
            "start_mode": StartMode.JUMP,
            "end_thickness": 1,
            "relative_angle": direction_range[0],
            "bounding_size": (30, 30),
            "fill": True,
        }
        for attr, expected_value in expected_polygon_attrs.items():
            actual = getattr(segment, attr, None)
            assert actual == expected_value, f"IRREGULAR_POLYGON: Expected {attr} to be {expected_value}, got {actual}"
        # Check that corners and lines_list have 5 elements.
        assert hasattr(segment, "corners"), "IRREGULAR_POLYGON: Segment should have a 'corners' attribute."
        assert hasattr(segment, "lines_list"), "IRREGULAR_POLYGON: Segment should have a 'lines_list' attribute."
        assert isinstance(segment.corners,
                          (list, np.ndarray)), "IRREGULAR_POLYGON: corners should be a list or numpy array."
        assert len(segment.corners) == 5, f"IRREGULAR_POLYGON: Expected 5 corners, got {len(segment.corners)}"
        assert isinstance(segment.lines_list, list), "IRREGULAR_POLYGON: lines_list should be a list."
        assert len(
            segment.lines_list) == 5, f"IRREGULAR_POLYGON: Expected 5 line segments, got {len(segment.lines_list)}"

