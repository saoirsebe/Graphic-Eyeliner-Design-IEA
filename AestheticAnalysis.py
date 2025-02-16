import math

from matplotlib import pyplot as plt
from A import SegmentType, StartMode, get_quadratic_points
from EyelinerWingGeneration import  generate_eyeliner_curve_lines
import numpy as np
from scipy.interpolate import interp1d


def check_points_left(points_array, is_left, threshold=0.8, x_limit=3):
    """Check if at least `threshold` percent of points are on the left of x_limit."""
    x_coords = points_array[:, 0]

    if is_left:
        side_count = np.sum(x_coords <= x_limit)
    else:
        side_count = np.sum(x_coords >= x_limit)

    # Calculate the percentage of points on the specified side
    total_points = len(x_coords)
    percentage = side_count / total_points
    # Check if the percentage is greater than or equal to the threshold
    return percentage >= threshold

def resample_curve(points, num_resize_val):
    """Resample a curve to have exactly num_samples points."""
    distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    total_distance = cumulative_distances[-1]
    uniform_distances = np.linspace(0, total_distance, num_resize_val)
    interp_x = interp1d(cumulative_distances, points[:, 0], kind='linear')
    interp_y = interp1d(cumulative_distances, points[:, 1], kind='linear')
    return np.vstack((interp_x(uniform_distances), interp_y(uniform_distances))).T

def resample_directions_or_curvatures(values, points, num_resize_val):
    """Resample directions or curvatures arrays based on the distance between points:
        if distance is < threshold between consecutive points then calculate distance between point and next point with distance > threshold instead."""
    # Compute distances between successive points
    distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0,0)  # Insert a 0 at the start to match the number of 'values'

    # Create uniform distances for resampling
    total_distance = cumulative_distances[-1]
    uniform_distances = np.linspace(0, total_distance, num_resize_val)
    # Interpolate values (curvatures or directions) based on cumulative distances
    if len(values)==len(cumulative_distances) - 2:
        cumulative_distances = cumulative_distances[1:-1]
    elif len(values)==len(cumulative_distances) - 1:
        cumulative_distances=cumulative_distances[1:]
    else:
        raise ValueError("The length of 'values' must be one/two less than the length of 'points'")
    #cumulative_distances_unique = cumulative_distances[0]
    cumulative_distances_unique, unique_indices = np.unique(cumulative_distances, return_index=True)
    values_unique = values[unique_indices]
    if cumulative_distances_unique.size == 0:
        print("cumulative_distances:", cumulative_distances)
        print("distances:", distances)
        print("points:", points)
        print("values:", values)
        raise ValueError("cumulative_distances_unique array must not be empty.")
    if values_unique.size == 0:
        print("values:", values)
        raise ValueError("values_unique array must not be empty.")
    if cumulative_distances_unique.shape[0] != values_unique.shape[0]:
        raise ValueError("Input arrays must have the same length.")

    interp_values = interp1d(cumulative_distances_unique, values_unique, kind='linear',axis=0, bounds_error=False,fill_value="extrapolate")
    #print("NaN in cumulative_distances:", np.isnan(cumulative_distances).any())

    #print("Cumulative Distances Range:", min(cumulative_distances), max(cumulative_distances))
    #print("Uniform Distances Range:", min(uniform_distances), max(uniform_distances))
    #print(not np.all(np.diff(cumulative_distances) > 0))
    #extrapolated_values = interp_values([min(cumulative_distances) - 1, max(cumulative_distances) + 1])
    #print(extrapolated_values)

    return interp_values(uniform_distances)

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

def direction_between_points(point1,point2):
    tangent = np.diff((point1, point2), axis=0)
    # Prevent division by zero when normalizing
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8  # Prevent zero vectors by setting a small value
    return tangent / norms

def compute_directions(points):
    directions = []
    for i in range(len(points)-1):#1 less?
        val = i
        next_val = i +1
        distance_from_next_point = np.linalg.norm(np.array(points[val]) - np.array(points[next_val]))
        while distance_from_next_point <= 0.0001:
            next_val +=1
            if next_val <len(points):
                distance_from_next_point = np.linalg.norm(np.array(points[val]) - np.array(points[next_val]))
            else:
                val = i-1
                next_val =i
                while distance_from_next_point <= 0.0001:
                    val -= 1
                    distance_from_next_point = np.linalg.norm(np.array(points[val]) - np.array(next_val))
        directions.append(direction_between_points(points[val], points[next_val]))
    return np.array(directions)

def compair_overlapping_sections(overlapping_points_segment, overlapping_points_eye_shape):
    """Calculates curvature and direction similarities between two sections (points arrays)"""
    bezier_curvature = calculate_curvature(overlapping_points_segment)
    eye_curvature = calculate_curvature(overlapping_points_eye_shape)
    # Compute shape similarity (curvature comparison)
    num_resize = max(len(overlapping_points_segment), len(overlapping_points_eye_shape))
    if len(bezier_curvature) <= 1:
        print("len(bezier_curvature) <=1:")
        print("overlapping_points_bezier:", overlapping_points_segment)
        shape_similarity = 0
    else:
        bezier_curvature_resampled = resample_directions_or_curvatures(bezier_curvature, overlapping_points_segment,
                                                                       num_resize)
        eye_curvature_resampled = resample_directions_or_curvatures(eye_curvature, overlapping_points_eye_shape,
                                                                    num_resize)
        # shape_similarity = 1 - np.mean(np.abs(bezier_curvature - eye_curvature))
        shape_similarity = 1 - np.sqrt(np.mean((bezier_curvature_resampled - eye_curvature_resampled) ** 2))

    segment_directions = compute_directions(overlapping_points_segment)
    eye_curve_directions = compute_directions(overlapping_points_eye_shape)

    if segment_directions.size == 0:
        print("bezier_directions.size == 0")
        print("overlapping_points_bezier", overlapping_points_segment)
        direction_similarity = 0
    else:
        # print("eye_curve_directions",eye_curve_directions)
        num_resize = max(len(overlapping_points_segment), len(overlapping_points_eye_shape))
        bezier_directions_resampled = resample_directions_or_curvatures(segment_directions, overlapping_points_segment,
                                                                        num_resize)
        eye_directions_resampled = resample_directions_or_curvatures(eye_curve_directions, overlapping_points_eye_shape,
                                                                     num_resize)
        # print("eye_directions_resampled",eye_directions_resampled)
        # Compute direction similarity (angle between normalized tangent vectors)
        dot_products = np.sum(bezier_directions_resampled * eye_directions_resampled, axis=1)
        direction_similarity = np.mean(dot_products)
    return shape_similarity, direction_similarity

def compare_with_eyelid_curves(bezier_points, eye_points, is_upper,num_samples=100):
    """
    Compare the curvature and direction of a segment and an eyelid curve curve.

    Returns:
        dict: A dictionary containing shape similarity, direction similarity, and overall similarity.
        directional and curvature similarity values have upper bounds of 1 if arrays are identical
    """
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

    # Resample both curves to have the same number of points
    num_resize = max(len(bezier_points) , len(eye_points))
    bezier_position_resampled = resample_curve(bezier_points, num_resize)
    eye_position_resampled = resample_curve(eye_points, num_resize)

    position_score = apply_upper_lower_condition(bezier_position_resampled, eye_position_resampled, is_upper)
    if position_score == 0:
        return{
            "shape_similarity": 0,
            "direction_similarity": 0,
            "overall_similarity": 0
        }

    overlapping_points_segment, overlapping_points_eye_shape = get_overlapping_points(bezier_points, eye_points)

    if overlapping_points_segment.shape[0]>4 and overlapping_points_eye_shape.shape[0]>4:
        shape_similarity, direction_similarity= compair_overlapping_sections(overlapping_points_segment, overlapping_points_eye_shape)
    else:
        shape_similarity = 0
        direction_similarity = 0
    overlap_length = math.sqrt((overlapping_points_segment[-1][0] - overlapping_points_segment[0][0]) ** 2 + (
                    overlapping_points_segment[-1][1] - overlapping_points_segment[0][1]) ** 2)
    # Combine into overall similarity
    overall_similarity = (shape_similarity + direction_similarity) / 2
    return {
        "shape_similarity": shape_similarity,
        "direction_similarity": direction_similarity,
        "overall_similarity": overall_similarity
    } ,overlap_length

def score_segment_against_eyelid_shape(segment, upper_curve, lower_curve, tolerance=0.1):
    points = segment.points_array
    """Runs compare_with_eyelid_curves for upper eyelid and lower eyelid curves, returning the score based on the overall similarity values returned"""

    # Calculate curvature of the segment
    upper_curve_results , overlap_length = compare_with_eyelid_curves(points,upper_curve,True, num_samples=100)
    lower_curve_results , overlap_length = compare_with_eyelid_curves(points,lower_curve,False, num_samples=100)
    #print("upper_curve_match", upper_curve_results)
    #print("lower_curve_match", lower_curve_results)

    # Assign scores
    score = 0
    if upper_curve_results["overall_similarity"]>0.6:
        score+= overlap_length * upper_curve_results["overall_similarity"]
    elif lower_curve_results["overall_similarity"]>0.6:
        score += overlap_length * lower_curve_results["overall_similarity"]
    else:
        score -= 0.25  # Penalty for deviating

    return score

def compair_segment_wing_shape(segment, curve1, curve2):
    points = segment.points_array
    overall_similarity1 = 0
    overall_similarity2 = 0

    overlapping_points_segment_1, overlapping_points_curve1 = get_overlapping_points(points, curve1)
    overlapping_points_segment_2, overlapping_points_curve2 = get_overlapping_points(points, curve2)
    if overlapping_points_segment_1.shape[0] > 4 and overlapping_points_curve1.shape[0] > 4:
        shape_similarity1, direction_similarity1 = compair_overlapping_sections(overlapping_points_segment_1, overlapping_points_curve1)
        overall_similarity1 = (shape_similarity1 + direction_similarity1) / 2
    if overlapping_points_segment_2.shape[0] > 4 and overlapping_points_curve2.shape[0] > 4:
        shape_similarity2, direction_similarity2 = compair_overlapping_sections(overlapping_points_segment_2, overlapping_points_curve2)
        overall_similarity2 = (shape_similarity2 + direction_similarity2) / 2
    best_overall = max(overall_similarity1, overall_similarity2)

    if best_overall>0.6:
        if best_overall == overall_similarity1:
            length = math.sqrt((overlapping_points_segment_1[-1][0] - overlapping_points_segment_1[0][0]) ** 2 + (
                    overlapping_points_segment_1[-1][1] - overlapping_points_segment_1[0][1]) ** 2)
            best_overall=best_overall * length
        if best_overall == overall_similarity2:
            length = math.sqrt((overlapping_points_segment_2[-1][0] - overlapping_points_segment_2[0][0]) ** 2 + (
                    overlapping_points_segment_2[-1][1] - overlapping_points_segment_2[0][1]) ** 2)
            best_overall = best_overall * length
    else:
        best_overall = -0.25  # Penalty for deviating
    return best_overall


def analyse_design_shapes(design):
    """
    Analyse the entire design and calculate a total score.
    """
    upper_x, upper_y = get_quadratic_points(-0.5, 0, 1, -1, 1)
    lower_x, lower_y = get_quadratic_points(0.5, 0, 0, -1, 1)
    upper_curve = np.column_stack(([x * 3 for x in upper_x], [y * 3 for y in upper_y]))
    lower_curve = np.column_stack(([x * 3 for x in lower_x], [y * 3 for y in lower_y]))

    total_score = 0
    segments = design.get_all_nodes()
    for segment in segments:
        if segment.segment_type == SegmentType.LINE:
            if check_points_left(segment.points_array,True):
                #If 80% of segment is left of the eye corner then compare segment with eyelid curve
                total_score += segment.length * score_segment_against_eyelid_shape(segment, upper_curve, lower_curve)
            elif check_points_left(segment.points_array,False):
                # If 80% of segment is right of the eye corner then compare segment with wing shape curves
                eyeliner_curve1, eyeliner_curve2 = generate_eyeliner_curve_lines()
                total_score += segment.length * compair_segment_wing_shape(segment, eyeliner_curve1, eyeliner_curve2)

        if segment.segment_type == SegmentType.IRREGULAR_POLYGON:
            max_score = 0
            #Adds score of line in polygon with the highest similarity with eyelid/eyeliner shape
            for line in segment.lines_list:
                points_array = line.points_array
                alignment_score = 0
                if check_points_left(points_array, True):
                    # If 80% of segment is left of the eye corner then compare segment with eyelid curve
                    alignment_score = score_segment_against_eyelid_shape(line, upper_curve,lower_curve)
                elif check_points_left(points_array, False):
                    # If 80% of segment is right of the eye corner then compare segment with wing shape curves
                    eyeliner_curve1, eyeliner_curve2 = generate_eyeliner_curve_lines()
                    alignment_score = compair_segment_wing_shape(line, eyeliner_curve1, eyeliner_curve2)

                if alignment_score >  max_score:
                    max_score = alignment_score
            total_score += max_score

    return total_score


"""
random_design = EyelinerDesign()
new_segment = create_segment(
        segment_type=SegmentType.LINE,
        start = (0,3),
        start_mode=StartMode.CONNECT,
        length=3,
        relative_angle=0,
        start_thickness=2.5,
        end_thickness=1,
        colour="red",
        curviness= 0.5 ,
        curve_direction=0.5,
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
        relative_angle=70,
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
"""


