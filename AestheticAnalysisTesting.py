import numpy as np
import pytest

from AestheticAnalysis import *
from Segments import StarSegment


def test_resample_curve():
    # Create a simple curve
    points = np.array([[0, 0], [1, 2], [3, 3], [4, 5]])
    num_resize_val = 10

    resampled_points = resample_curve(points, num_resize_val)

    # Ensure correct shape
    assert resampled_points.shape == (num_resize_val, 2)

    # Ensure points are still within bounds
    assert np.min(points[:, 0]) <= np.min(resampled_points[:, 0]) <= np.max(points[:, 0])
    assert np.min(points[:, 1]) <= np.min(resampled_points[:, 1]) <= np.max(points[:, 1])
    assert np.min(points[:, 0]) <= np.max(resampled_points[:, 0]) <= np.max(points[:, 0])
    assert np.min(points[:, 1]) <= np.max(resampled_points[:, 1]) <= np.max(points[:, 1])

    print("Resample curve test passed successfully!")


def test_resample_curvatures():
    curvature = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    num_points = 10

    resampled_curvatures = resample_curvatures(curvature, num_points)

    # Ensure correct length
    assert len(resampled_curvatures) == num_points

    # Ensure values are within the original range
    assert np.min(curvature) <= np.min(resampled_curvatures) <= np.max(curvature)
    assert np.min(curvature) <= np.max(resampled_curvatures) <= np.max(curvature)

    # Check interpolation accuracy
    expected_curvatures = np.interp(np.linspace(0, 1, num_points), np.linspace(0, 1, len(curvature)), curvature)
    assert np.allclose(resampled_curvatures, expected_curvatures, atol=1e-6)

    print("Resample curvatures test passed successfully!")


@pytest.mark.parametrize(
    "points, expected_curvatures", [
        # Case 1: 5 points
        (np.array([[0, 0], [1, 1], [2, 0], [3, -1], [4, 0]]), np.array([-np.pi / 2, 0, np.pi / 2])),

        # Case 2: A horizontal line (flat, curvature should be 0)
        (np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]), np.array([0, 0, 0])),

        # Case 3: Array where first two points are equal
        (np.array([[0, 0], [0, 0], [1, 2], [2, 3], [-1, 3]]), np.array([1.1071487177940904,-0.32175055439664213,2.356194490192345])),

        # Case 4: Array where three points are equal
        (np.array([[0, 0], [1, 0], [1, 0], [1, 0], [1, 1],[-1, 2],[-2, 4]]), np.array([1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 1.1071487177940904, -0.6435011087932843])),

        # Case 5: Array where two points are very similar
        (np.array([[80.33, 98.753], [80.153, 98.769], [79.976, 98.784], [79.799, 98.8], [79.799, 98.80006],[79.622, 98.816]]), np.array([0.005606707381364373,-0.005606707381284437,-7.238654120556021e-14, 0.0006724710849650428]))
    ]
)

def test_calculate_curvature(points, expected_curvatures):
    curvatures = calculate_curvature(points)

    # Assert correct length
    assert len(curvatures) == len(points) - 2

    # Compare calculated curvatures with expected ones
    assert np.allclose(curvatures, expected_curvatures, atol=1e-6)


import numpy as np
import pytest


# Assuming the check_points_left function is already defined

@pytest.mark.parametrize(
    "points, x_limit, is_left, threshold, expected_result",
    [
        # Test case 1: 80% of points should be on the left
        (np.array([[0.5, 1], [0.3, 2], [0.6, 3], [0.4, 4], [0.7, 5]]), 0.6, True, 0.8, True),

        # Test case 2: 80% of points should be on the right, only 60% are
        (np.array([[0.5, 1], [0.3, 2], [0.6, 3], [0.6, 4], [0.7, 5]]), 0.6, False, 0.8, False),

        # Test case 2: 80% of points should be on the right, 80% are
        (np.array([[0.5, 1], [0.6, 2], [0.6, 3], [0.7, 4], [0.8, 5]]), 0.6, False, 0.8, True),

        # Test case 3: 80% of points should be on the left, 60% are
        (np.array([[0.5, 1], [0.3, 2], [0.6, 3], [0.7, 4], [0.8, 5]]), 0.6, True, 0.8, False),

        # Test case 4: 80% of points should be on the left, 100% are
        (np.array([[0.5, 1], [0.3, 2], [0.4, 3]]), 0.6, True, 0.8, True),

        # Test case 5: Empty array should return False
        (np.array([]), 0.6, True, 0.8, False),
    ]
)
def test_check_points_left(points, x_limit, is_left, threshold, expected_result):
    result = check_points_left(points, is_left, threshold, x_limit)
    assert result == expected_result, f"Failed for points: {points}, expected: {expected_result}, got: {result}"


@pytest.mark.parametrize(
    "curve1, curve2, expected_curve1_overlap, expected_curve2_overlap",
    [
        # Test case 1: Partial overlap between the curves
        (
            np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
            np.array([[2.0, 200.0], [3.0, 300.0], [4.0, 400.0]]),
            np.array([[2.0, 20.0], [3.0, 30.0]]),
            np.array([[2.0, 200.0], [3.0, 300.0]])
        ),
        # Test case 2: Complete overlap (both curves share the same x-range)
        (
            np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
            np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]]),
            np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
            np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
        ),
        # Test case 3: No overlap between the x ranges
        (
            np.array([[1.0, 10.0], [2.0, 20.0]]),
            np.array([[3.0, 300.0], [4.0, 400.0]]),
            np.array([]).reshape(0, 2),
            np.array([]).reshape(0, 2)
        ),
        # Test case 4: Overlap only at the boundary (edge-case)
        (
            np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
            np.array([[3.0, 300.0], [4.0, 400.0]]),
            np.array([[3.0, 30.0]]),
            np.array([[3.0, 300.0]])
        ),
    ]
)
def test_get_overlapping_points(curve1, curve2, expected_curve1_overlap, expected_curve2_overlap):
    curve1_overlap, curve2_overlap = get_overlapping_points(curve1, curve2)
    np.testing.assert_array_equal(curve1_overlap, expected_curve1_overlap)
    np.testing.assert_array_equal(curve2_overlap, expected_curve2_overlap)


@pytest.mark.parametrize("point1, point2, expected", [
    # Basic horizontal direction
    ([0, 0], [1, 0], [1, 0]),
    # Diagonal: difference is [1, 2]
    ([1, 1], [2, 3], [1/np.sqrt(5), 2/np.sqrt(5)]),
    # Identical points: expect zero vector
    ([1, 1], [1, 1], [0, 0]),
])
def test_direction_between_points(point1, point2, expected):
    result = direction_between_points(point1, point2)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)

@pytest.mark.parametrize("points, expected_angle", [
    # Horizontal line: 0 rad
    (np.array([[0, 0], [1, 0]]), 0.0),
    # Vertical line: pi/2 rad
    (np.array([[0, 0], [0, 1]]), np.pi/2),
    # Diagonal line: from (1,2) to (4,6) → dx=3, dy=4, angle = arctan2(4,3)
    (np.array([[1, 2], [4, 6]]), np.arctan2(4, 3)),
])
def test_compute_global_direction(points, expected_angle):
    result = compute_global_direction(points)
    assert np.allclose(result, expected_angle, rtol=1e-5, atol=1e-8)

@pytest.mark.parametrize("points, expected_directions", [
    # Two segments in a horizontal line: both directions should be [1, 0]
    (np.array([[0, 0], [1, 0], [2, 0]]),
     np.array([[1, 0], [1, 0]])),
    # Two segments in a 45° diagonal line: direction should be [1/sqrt2, 1/sqrt2]
    (np.array([[0, 0], [1, 1], [2, 2]]),
     np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), 1/np.sqrt(2)]])),
])
def test_compute_directions(points, expected_directions):
    result = compute_directions(points)
    np.testing.assert_allclose(result, expected_directions, rtol=1e-5, atol=1e-8)

@pytest.mark.parametrize("v1, v2, expected", [
    # Same vector: cosine similarity = 1
    ([1, 0], [1, 0], 1.0),
    # Orthogonal vectors: cosine similarity = 0
    ([1, 0], [0, 1], 0.0),
    # Opposite direction: cosine similarity = -1
    ([1, 1], [-1, -1], -1.0),
])
def test_cosine_similarity(v1, v2, expected):
    result = cosine_similarity(np.array(v1), np.array(v2))
    assert np.allclose(result, expected, rtol=1e-5, atol=1e-8)

@pytest.mark.parametrize("v1, v2, expected_angle", [
    # Same vector: angle = 0
    ([1, 0], [1, 0], 0.0),
    # Perpendicular vectors: angle = pi/2
    ([1, 0], [0, 1], np.pi/2),
    # Opposite vectors: angle = pi
    ([1, 1], [-1, -1], np.pi),
    # One vector zero: angle = 0 (by definition)
    ([0, 0], [1, 0], 0.0),
])
def test_angle_between_vectors(v1, v2, expected_angle):
    result = angle_between_vectors(np.array(v1), np.array(v2))
    assert np.allclose(result, expected_angle, rtol=1e-5, atol=1e-8)

@pytest.mark.parametrize(
    "radius, arm_length, expected",
    [
        # Valid: arm_length is sufficiently large compared to radius, and total size < 20.
        (4, 5, True),
        # Invalid: arm_length too short relative to radius (5 * 0.4 = 2, but arm_length is 1)
        (5, 1, False),
        # Invalid: radius too large compared to arm_length (arm_length * 1.5 = 4.5, but radius is 7)
        (7, 4, False),
        # Invalid: total size exceeds 20 (12 + 10 = 22)
        (12, 10, False),
        # Boundary condition: for radius=10 and arm_length=4, 10 > 4*1.5 (i.e. 10 > 6) so should be invalid.
        (10, 4, False),
        # Boundary condition: total size exactly 20 (radius + arm_length == 20) is valid.
        (10, 10, True),
        # Boundary condition: radius exactly 1.5 times arm_length is valid (e.g., 9 vs. 6: 9 == 6*1.5).
        (9, 6, True),
        # Slightly above boundary: radius greater than arm_length * 1.5 should fail (9.1 > 9).
        (9.1, 6, False),
    ]
)
def test_validate_star_parameters(radius, arm_length, expected):
    star = StarSegment(
        segment_type=SegmentType.STAR,
        start=(0, 0),
        colour="red",
        star_type="dummy",
        radius=radius,
        arm_length=arm_length,
        num_points=5,
        asymmetry=0,
        start_mode=0,
        end_thickness=1,
        relative_angle=0,
        fill=False
    )
    result = validate_star_parameters(star)
    assert result == expected, f"For star with radius={radius} and arm_length={arm_length}, expected {expected} but got {result}"