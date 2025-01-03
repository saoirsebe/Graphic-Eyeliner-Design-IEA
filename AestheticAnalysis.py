import numpy as np
import matplotlib.pyplot as plt
from A import SegmentType, StartMode
from EyelinerWingGeneration import get_quadratic_points, generate_eye_curve_directions
from EyelinerDesign import random_gene, EyelinerDesign

import numpy as np
from scipy.interpolate import interp1d

from Segments import create_segment


def compare_curves(bezier_points, eye_points, eye_curve_shape, is_upper,num_samples=100):
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
    eye_curve_resampled = resample_curve(eye_curve_shape, num_samples)
    eye_resampled = resample_curve(eye_points, num_samples)

    def apply_upper_lower_condition(bezier_curve, quadratic_curve, is_upper, threshold=0.9):
        count_valid = 0
        total_points = len(bezier_curve)

        for bezier_point, quadratic_point in zip(bezier_curve, quadratic_curve):
            if is_upper:
                # Bezier should be above the quadratic curve
                if bezier_point[1] > quadratic_point[1]:
                    count_valid += 1
            else:
                # Bezier should be below the quadratic curve
                if bezier_point[1] < quadratic_point[1]:
                    count_valid += 1

        # If 90% or more of the points are valid (either above or below), return 1, otherwise return 0
        if count_valid / total_points >= threshold:
            return 1
        else:
            return 0

    position_score = apply_upper_lower_condition(bezier_resampled, eye_resampled, is_upper)
    if position_score == 0:
        return{
            "shape_similarity": 0,
            "direction_similarity": 0,
            "overall_similarity": 0
        }

    # Compute tangent vectors and normalize to get directions
    def compute_directions(points):
        tangents = np.diff(points, axis=0)
        # Prevent division by zero when normalizing
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8  # Prevent zero vectors by setting a small value
        directions = tangents / norms
        return directions

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

    overlapping_points_bezier, overlapping_points_eye_shape = get_overlapping_points(bezier_resampled, eye_curve_resampled)

    if overlapping_points_bezier.shape[0]>0 and overlapping_points_eye_shape.shape[0]>0:
        # Plot the curves
        plt.figure(figsize=(8, 6))
        plt.plot(overlapping_points_bezier[:, 0], overlapping_points_bezier[:, 1], label='Bezier Curve', color='b')
        plt.plot(overlapping_points_eye_shape[:, 0], overlapping_points_eye_shape[:, 1], label=f'Quadratic Curve {is_upper}',color='g')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')  # Equal aspect ratio ensures that arrows are displayed proportionally
        # Show plot
        plt.show()

        bezier_curvature = calculate_curvature(overlapping_points_bezier)
        eye_curvature = calculate_curvature(overlapping_points_eye_shape)

        # Compute shape similarity (curvature comparison)

        #shape_similarity = 1 - np.mean(np.abs(bezier_curvature - eye_curvature))
        shape_similarity = 1 - np.sqrt(np.mean((bezier_curvature - eye_curvature) ** 2))

        bezier_directions = compute_directions(overlapping_points_bezier)
        eye_curve_directions = compute_directions(overlapping_points_eye_shape)

        # Compute direction similarity (angle between normalized tangent vectors)
        dot_products = np.sum(bezier_directions * eye_curve_directions, axis=1)
        direction_similarity = np.mean(dot_products)
    else:
        shape_similarity = 0
        direction_similarity = 0

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
    top_eye_curve, bottom_eye_curve = generate_eye_curve_directions()
    upper_curve_results= compare_curves(points,upper_curve,top_eye_curve,True, num_samples=100)
    lower_curve_results= compare_curves(points,lower_curve,bottom_eye_curve,False, num_samples=100)
    print("upper_curve_match", upper_curve_results)
    print("lower_curve_match", lower_curve_results)

    # Assign scores
    score = 0
    if upper_curve_results["overall_similarity"]>0:
        score+=upper_curve_results["overall_similarity"]
    elif lower_curve_results["overall_similarity"]>0:
        score += lower_curve_results["overall_similarity"]
    else:
        score -= 2  # Penalty for deviating

    return score


def analyze_design(design):
    """
    Analyze the entire design and calculate a total score.
    """
    upper_x, upper_y = get_quadratic_points(-0.5, 0, 1, -1, 1)
    lower_x, lower_y = get_quadratic_points(0.5, 0, 0, -1, 1)
    upper_curve = np.column_stack(([x * 3 for x in upper_x], [y * 3 for y in upper_y]))
    lower_curve = np.column_stack(([x * 3 for x in lower_x], [y * 3 for y in lower_y]))

    # Define wing direction
    wing_direction = [1, 0.2]

    total_score = 0
    for segment in design.segments:
        if segment.segment_type == SegmentType.LINE:
            total_score += score_segment(segment, upper_curve, lower_curve, wing_direction)
    return total_score

#random_design = random_gene(0)
random_design = EyelinerDesign()
new_segment = create_segment(
        segment_type=SegmentType.LINE,
        start = (0,3),
        start_mode=StartMode.CONNECT,
        length=3,
        relative_angle=40,
        start_thickness=2.5,
        end_thickness=1,
        colour="red",
        curviness= 0 ,
        curve_direction=0.2,
        curve_location=0.5,
        start_location=0.6,
        split_point=0.2

)
random_design.add_segment(new_segment)
new_segment = create_segment(
        segment_type=SegmentType.LINE,
        start = (1,-1.5),
        start_mode=StartMode.JUMP,
        length=1.5,
        relative_angle=110,
        start_thickness=6,
        end_thickness=4,
        colour="orange",
        curviness= 0.5 ,
        curve_direction=20,
        curve_location=0.5,
        start_location=0.6,
        split_point=0

)
random_design.add_segment(new_segment)
new_segment = create_segment(
        segment_type=SegmentType.LINE,
        start = (-2,3),
        start_mode=StartMode.JUMP,
        length=7,
        relative_angle=90,
        start_thickness=2.5,
        end_thickness=1,
        colour="pink",
        curviness= 0 ,
        curve_direction=90,
        curve_location=0.5,
        start_location=0.6,
        split_point=0

)
random_design.add_segment(new_segment)
fig = random_design.render()
fig.show()
score = analyze_design(random_design)
print("Score:", score)
