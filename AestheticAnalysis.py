import numpy as np
import matplotlib.pyplot as plt
from A import SegmentType
from EyelinerWingGeneration import get_quadratic_points
from EyelinerDesign import random_gene

import numpy as np
from scipy.interpolate import interp1d


def compare_curves(bezier_points, eye_points, num_samples=100):
    """
    Compare the curvature and direction of a Bezier curve and a quadratic curve.

    Returns:
        dict: A dictionary containing shape similarity, direction similarity, and overall similarity.
    """

    def resample_curve(points, num_samples):
        """Resample a curve to have exactly num_samples points."""
        distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
        total_distance = cumulative_distances[-1]
        uniform_distances = np.linspace(0, total_distance, num_samples)
        interp_x = interp1d(cumulative_distances, points[:, 0], kind='linear')
        interp_y = interp1d(cumulative_distances, points[:, 1], kind='linear')
        return np.vstack((interp_x(uniform_distances), interp_y(uniform_distances))).T

    # Resample both curves to have the same number of points
    bezier_resampled = resample_curve(bezier_points, num_samples)
    eye_resampled = resample_curve(eye_points, num_samples)

    # Compute tangent vectors and normalize to get directions
    def compute_directions(points):
        tangents = np.diff(points, axis=0)
        directions = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)
        return directions

    bezier_directions = compute_directions(bezier_resampled)
    quadratic_directions = compute_directions(eye_resampled)

    # Compute curvature for both sets of points
    def calculate_curvature(points):
        vectors = np.diff(points, axis=0)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        curvatures = np.diff(angles)
        return curvatures

    def get_overlapping_points(curve1, curve2, tolerance=0.1):
        overlapping_points_curve1 = []
        overlapping_points_curve2 = []

        for point1 in curve1:
            # Find points in curve2 with x-value within the tolerance range
            mask = np.abs(curve2[:, 0] - point1[0]) <= tolerance
            if np.any(mask):
                # Get the corresponding point(s) from curve2
                overlapping_points_curve2.append(curve2[mask][0])  # Just take the first match
                overlapping_points_curve1.append(point1)

        return np.array(overlapping_points_curve1), np.array(overlapping_points_curve2)

    overlapping_points_bezier, overlapping_points_eye = get_overlapping_points(bezier_resampled, eye_resampled)

    # Plot the curves
    plt.figure(figsize=(8, 6))
    plt.plot(overlapping_points_bezier[:, 0], overlapping_points_bezier[:, 1], label='Bezier Curve', color='b')
    plt.plot(overlapping_points_eye[:, 0], overlapping_points_eye[:, 1], label='Quadratic Curve', color='g')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Equal aspect ratio ensures that arrows are displayed proportionally
    # Show plot
    plt.show()

    bezier_curvature = calculate_curvature(overlapping_points_bezier)
    eye_curvature = calculate_curvature(overlapping_points_eye)

    # Compute shape similarity (curvature comparison)
    shape_similarity = 1 - np.mean(np.abs(bezier_curvature - eye_curvature))

    # Compute direction similarity (angle between normalized tangent vectors)
    dot_products = np.sum(bezier_directions * quadratic_directions, axis=1)
    direction_similarity = np.mean(dot_products)

    # Combine into overall similarity
    overall_similarity = (shape_similarity + direction_similarity) / 2

    return {
        "shape_similarity": shape_similarity,
        "direction_similarity": direction_similarity,
        "overall_similarity": overall_similarity
    }


def score_segment(segment, upper_curve, lower_curve, wing_direction, tolerance=0.1):
    """
    Score a single segment based on how well it aligns with the natural curves.
    """
    points = segment.points_array

    # Calculate curvature of the segment
    upper_curve_match= compare_curves(points,upper_curve, num_samples=100)
    lower_curve_match= compare_curves(points,lower_curve, num_samples=100)
    print("upper_curve_match", upper_curve_match)
    print("lower_curve_match", lower_curve_match)

    # Check if it's a wing shape (straight or slightly curved outward)
    is_wing = False
    """
    wing_vector = np.array(wing_direction)
    segment_vector = end - start
    dot_product = np.dot(wing_vector, segment_vector) / (np.linalg.norm(wing_vector) * np.linalg.norm(segment_vector))
    if dot_product > 0.95:  # Angle close to 0 (aligned with wing direction)
        is_wing = True
    """
    # Assign scores
    score = 0
    if upper_curve_match or lower_curve_match:
        score += 5  # High score for matching natural curves
    if is_wing:
        score += 5  # Bonus for good wing shape
    if not upper_curve_match and not lower_curve_match and not is_wing:
        score -= 2  # Penalty for deviating

    return score


def analyze_design(design):
    """
    Analyze the entire design and calculate a total score.
    """
    upper_x, upper_y = get_quadratic_points(-0.5, 0, 1, -1, 1)
    lower_x, lower_y = get_quadratic_points(0.5, 0, 0, -1, 1)

    # Scale curves
    upper_curve = np.column_stack(([x * 3 for x in upper_x], [y * 3 for y in upper_y]))
    lower_curve = np.column_stack(([x * 3 for x in lower_x], [y * 3 for y in lower_y]))

    # Define wing direction
    wing_direction = [1, 0.2]

    total_score = 0
    for segment in design.segments:
        if segment.segment_type == SegmentType.LINE:
            total_score += score_segment(segment, upper_curve, lower_curve, wing_direction)
    return total_score

random_design = random_gene(0)
fig = random_design.render()
fig.show()
analyze_design(random_design)
