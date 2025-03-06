import math

from matplotlib import pyplot as plt
from A import SegmentType, StartMode, get_quadratic_points, eye_corner_start, upper_eyelid_coords, lower_eyelid_coords, \
    eyeliner_curve1, eyeliner_curve2, middle_curve_upper, middle_curve_lower
import numpy as np
from scipy.interpolate import interp1d


def check_points_left(points_array, is_left, threshold=0.8, x_limit=eye_corner_start[0]):
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

def direction_between_points(point1, point2):
    tangent = np.array(point2) - np.array(point1)
    norm = np.linalg.norm(tangent)  # Compute magnitude
    if norm == 0:  # Prevent division by zero
        return np.zeros_like(tangent)
    return tangent / norm

def compute_global_direction(points):
    """Compute the global direction from start to end of the curve."""
    dx = points[-1, 0] - points[0, 0]
    dy = points[-1, 1] - points[0, 1]
    return np.arctan2(dy, dx)


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

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def angle_between_vectors(v1, v2):
    """Compute the angle in radians between two vectors."""
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norms == 0:
        return 0  # Avoid division by zero
    return np.arccos(np.clip(dot_product / norms, -1.0, 1.0))

def compair_overlapping_sections(overlapping_points_segment, overlapping_points_eye_shape):
    """Calculates curvature and direction similarities between two sections (points arrays)"""
    if overlapping_points_segment[0][0]>overlapping_points_segment[-1][0]:
        overlapping_points_segment = overlapping_points_segment[::-1]
    if overlapping_points_eye_shape[0][0]>overlapping_points_eye_shape[-1][0]:
        overlapping_points_eye_shape = overlapping_points_eye_shape[::-1]

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

    segment_direction = compute_global_direction(overlapping_points_segment)#compute_directions(overlapping_points_segment)
    eye_curve_direction = compute_global_direction(overlapping_points_eye_shape)#compute_directions(overlapping_points_eye_shape)
    direction_difference = 1 - np.abs(segment_direction - eye_curve_direction)
    #print("direction_difference",direction_difference)
    """
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

        # Compute direction similarity
        similarities = [cosine_similarity(v1, v2) for v1, v2 in zip(bezier_directions_resampled, eye_directions_resampled)]
        direction_similarity  = np.mean(similarities)
        #direction_similarity = 1 - (mean_angle_difference / np.pi)
        def total_deflection(vectors):
            #Compute total angle change across a vector sequence.
            vectors = np.squeeze(vectors)  # Remove extra dimensions
            angle_changes = [angle_between_vectors(vectors[i], vectors[i + 1])
                             for i in range(len(vectors) - 1)]
            return np.sum(angle_changes)  # Sum total angle changes

        #eye_deflection = total_deflection(eye_directions_resampled)
        #bezier_deflection = total_deflection(bezier_directions_resampled)
        #print("eye_deflection:", eye_deflection)
        #print("bezier_deflection:", bezier_deflection)
        #direction_similarity = 1 - abs(eye_deflection - bezier_deflection) / max(eye_deflection, bezier_deflection)
    """

    return shape_similarity, direction_difference

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
        } , 0

    overlapping_points_segment, overlapping_points_eye_shape = get_overlapping_points(bezier_points, eye_points)

    if overlapping_points_segment.shape[0]>4 and overlapping_points_eye_shape.shape[0]>4:
        overlap_length = math.sqrt((overlapping_points_segment[-1][0] - overlapping_points_segment[0][0]) ** 2 + (
                overlapping_points_segment[-1][1] - overlapping_points_segment[0][1]) ** 2)
        if overlap_length>10:
            shape_similarity, direction_similarity = compair_overlapping_sections(overlapping_points_segment,overlapping_points_eye_shape)
        else:
            shape_similarity = 0
            direction_similarity = 0
            overlap_length = 0
    else:
        shape_similarity = 0
        direction_similarity = 0
        overlap_length = 0

    # Combine into overall similarity
    overall_similarity = (shape_similarity + direction_similarity) / 2
    return {
        "shape_similarity": shape_similarity,
        "direction_similarity": direction_similarity,
        "overall_similarity": overall_similarity
    } , overlap_length

def score_segment_against_eyelid_shape(segment, tolerance=0.1, to_print = False):
    upper_curve = upper_eyelid_coords
    lower_curve = lower_eyelid_coords
    points = segment.points_array
    """Runs compare_with_eyelid_curves for upper eyelid and lower eyelid curves, returning the score based on the overall similarity values returned"""

    # Calculate curvature of the segment
    upper_curve_results , upper_overlap_length = compare_with_eyelid_curves(points,upper_curve,True, num_samples=100)
    lower_curve_results , lower_overlap_length = compare_with_eyelid_curves(points,lower_curve,False, num_samples=100)

    # Assign scores
    score = 0
    if upper_curve_results["overall_similarity"]>0.7 and upper_curve_results["direction_similarity"]>0.7:# and upper_curve_results["direction_similarity"]>0.5:
        if to_print:
            print(f"for colour: {segment.colour} similarity to upper eyelid:", upper_curve_results["overall_similarity"])
            print("upper_curve_results[direction_similarity]", upper_curve_results["direction_similarity"])
            print("upper_curve_results[shape_similarity]", upper_curve_results["shape_similarity"])
        #print("overlap_length:", upper_overlap_length)
        score+= 1.7 * math.log(upper_overlap_length) * upper_curve_results["overall_similarity"]
    elif lower_curve_results["overall_similarity"]>0.7 and lower_curve_results["direction_similarity"]>0.7:
        if to_print:
            print(f"for colour: {segment.colour} similarity to lower eyelid:", lower_curve_results["overall_similarity"])
            print("lower_curve_results[direction_similarity]", lower_curve_results["direction_similarity"])
            print("lower_curve_results[shape_similarity]", lower_curve_results["shape_similarity"])
        #print("overlap_length:", lower_overlap_length)
        score += 1.5 * math.log(lower_overlap_length) * lower_curve_results["overall_similarity"]
    elif upper_curve_results["overall_similarity"] < 0.5 or lower_curve_results["direction_similarity"] < 0.4:
        if to_print:
            print("upper_curve_results[direction_similarity]", upper_curve_results["direction_similarity"])
            print("upper_curve_results[shape_similarity]", upper_curve_results["shape_similarity"])
        seg_array = segment.points_array
        seg_length = math.sqrt((seg_array[-1][0] - seg_array[0][0]) ** 2 + (seg_array[-1][1] - seg_array[0][1]) ** 2)
        score += -0.04 * seg_length  # Penalty for deviating

    return score

def check_overlap_length_then_similarity(overlapping_points_segment, overlapping_points_curve, to_print = False):
    overall_similarity = 0
    if overlapping_points_segment.shape[0] > 4 and overlapping_points_curve.shape[0] > 4:
        segment_overlap_length = math.sqrt((overlapping_points_segment[-1][0] - overlapping_points_segment[0][0]) ** 2 + (
                overlapping_points_segment[-1][1] - overlapping_points_segment[0][1]) ** 2)

        if segment_overlap_length>5:
            shape_similarity1, direction_similarity1 = compair_overlapping_sections(overlapping_points_segment, overlapping_points_curve)
            if to_print:
                print("shape_similarity1", shape_similarity1)
                print("direction_similarity1", direction_similarity1)
            overall_similarity = (shape_similarity1 + direction_similarity1) / 2
    else:
        segment_overlap_length = 0
    return overall_similarity , segment_overlap_length


def compair_middle_curve_shapes(segment, to_print = False):
    curve1 = middle_curve_upper
    curve2 = middle_curve_lower
    points = segment.points_array
    segment_overlap_length = 0
    overall_similarity = 0
    y_values = points[:, 1]
    # Check if all y values are above n
    all_above_n = np.all(y_values > eye_corner_start[1])
    if all_above_n:
        overlapping_points_segment_1, overlapping_points_curve1 = get_overlapping_points(points, curve1)
        overall_similarity , segment_overlap_length = check_overlap_length_then_similarity(overlapping_points_segment_1, overlapping_points_curve1, to_print)
    else:
        # Check if all y values are below n
        all_below_n = np.all(y_values < eye_corner_start[1])
        if all_below_n:
            overlapping_points_segment_2, overlapping_points_curve2 = get_overlapping_points(points, curve2)
            overall_similarity, segment_overlap_length = check_overlap_length_then_similarity(overlapping_points_segment_2, overlapping_points_curve2)

    if overall_similarity > 0.7:
        score_overall = 2 * overall_similarity * math.log(segment_overlap_length)
    elif overall_similarity < 0.6:
        seg_length = math.sqrt((points[-1][0] - points[0][0]) ** 2 + (
                points[-1][1] - points[0][1]) ** 2)
        score_overall = -0.02 * seg_length  # Penalty for deviating
    else:
        score_overall = 0
    return score_overall

def compair_segment_wing_shape(segment, curve1, curve2, to_print=False):
    points = segment.points_array

    overlapping_points_segment_1, overlapping_points_curve1 = get_overlapping_points(points, curve1)
    overlapping_points_segment_2, overlapping_points_curve2 = get_overlapping_points(points, curve2)
    overall_similarity1 , segment_1_overlap_length= check_overlap_length_then_similarity(overlapping_points_segment_1, overlapping_points_curve1, to_print = to_print)

    overall_similarity2 , segment_2_overlap_length= check_overlap_length_then_similarity(overlapping_points_segment_2, overlapping_points_curve2, to_print=to_print)
    best_overall = max(overall_similarity1, overall_similarity2)

    if best_overall>0.7:
        if best_overall == overall_similarity1:
            score= 2 * best_overall *  math.log(segment_1_overlap_length)
        elif best_overall == overall_similarity2:

            score = 2 * best_overall * math.log(segment_2_overlap_length)
        else:
            score = 0
    elif best_overall < 0.6:
        seg_array = segment.points_array
        seg_length = math.sqrt((seg_array[-1][0] - seg_array[0][0]) ** 2 + (seg_array[-1][1] - seg_array[0][1]) ** 2)
        score = -0.02 * seg_length  # Penalty for deviating
    else:
        score = 0
    return score


def check_points_middle(points_array):
    x_min = 90
    x_max = 140

    x_values = points_array[:, 0]
    in_range = (x_values >= x_min) & (x_values <= x_max)

    return np.sum(in_range) >= len(points_array) * 0.8 #Return if 80% of points are within this range


def analyse_design_shapes(design, to_print = False):
    """
    Analyse the entire design and calculate a total score.
    """
    n_of_stars = 0
    n_of_polygons = 0
    n_of_jumps = 0
    total_score = 0
    segments = design.get_all_nodes()
    for segment in segments:
        average_x = np.mean(segment.points_array[:, 0])
        if average_x <0:
            total_score -=5

        average_y = np.mean(segment.points_array[:, 1])
        if average_y < 50:
            total_score -= 2

        if segment.start_mode == StartMode.JUMP:
            n_of_jumps += 1

        if segment.segment_type == SegmentType.LINE:
            if segment.start_mode == StartMode.JUMP:
                total_score -=0.5
            elif segment.start_mode == StartMode.CONNECT_MID:
                total_score +=0.55
            elif segment.start_mode == StartMode.SPLIT:
                total_score +=0.55

            if check_points_left(segment.points_array,True):
                #If 80% of segment is left of the eye corner then compare segment with eyelid curve
                #print("checking against eyelid shape")
                line_score=score_segment_against_eyelid_shape(segment,to_print=to_print)
                left_score = line_score
                if check_points_middle(segment.points_array):
                    middle_score = compair_middle_curve_shapes(segment)

                    if middle_score > left_score:
                        if to_print:
                            print(f"line colour:{segment.colour} middle_score:", middle_score)
                        total_score += middle_score
                    else:
                        if to_print:
                            print(f"line colour:{segment.colour} score_segment_against_eyelid_shape:", line_score)
                        total_score += left_score
                else:
                    if to_print:
                        print(f"line colour:{segment.colour} score_segment_against_eyelid_shape:", line_score)
                    total_score += left_score

            elif check_points_left(segment.points_array,False):
                # If 80% of segment is right of the eye corner then compare segment with wing shape curves
                #print("checking against eyeliner shape")

                line_score = compair_segment_wing_shape(segment, eyeliner_curve1, eyeliner_curve2 )
                right_score = line_score
                if check_points_middle(segment.points_array):
                    middle_score = compair_middle_curve_shapes(segment)
                    middle_score = middle_score
                    if middle_score > right_score:
                        if to_print:
                            print(f"line colour:{segment.colour} middle_score:", middle_score)
                        total_score += middle_score
                    else:
                        if to_print:
                            print(f"line colour:{segment.colour} compair_segment_wing_shape:", line_score)
                        total_score += right_score
                else:
                    if to_print:
                        print(f"line colour:{segment.colour} compair_segment_wing_shape:", line_score)
                    total_score += right_score



        elif segment.segment_type == SegmentType.IRREGULAR_POLYGON:
            n_of_polygons +=1
            max_score = 0
            #Adds score of line in polygon with the highest similarity with eyelid/eyeliner shape

            #CHANGE TO JUST USE EDGE TO EDGE DIRECTIONS AND COMPAIR WITH WING DIRECTION
            """
            for line in segment.lines_list:
                points_array = line.points_array
                alignment_score = 0
                if check_points_left(points_array, True):
                    # If 80% of segment is left of the eye corner then compare segment with eyelid curve
                    alignment_score = score_segment_against_eyelid_shape(line)
                elif check_points_left(points_array, False):
                    # If 80% of segment is right of the eye corner then compare segment with wing shape curves

                    alignment_score = compair_segment_wing_shape(line, eyeliner_curve1, eyeliner_curve2)

                if alignment_score > max_score:
                    max_score = alignment_score
            print(f"polygon colour:{segment.colour} curve shape score:", max_score)
            total_score += max_score
            """

            if segment.is_eyeliner_wing:
                #Size of eyeliner wing polygon is 1 so size will be penalised
                total_score +=6
            else:
                # Score polygon based on size (bigger preferred on outside of eye and smaller nearer the inner corner)
                x_values = segment.points_array[:, 0]
                y_values = segment.points_array[:, 1]
                average_x = np.mean(x_values)
                average_y = np.mean(y_values)
                shape_size = max((max(x_values)-min(x_values)) , (max(y_values)-min(y_values)))
                if average_x <= 100:
                    k = 0.3  # Default for small x-values
                else:
                    k = 0.32 #+ 0.002 * (average_x - 100)  # Gradual increase for large x-values
                if average_y < 70 and average_x < 110:
                    k *= 0.7

                deviation = abs(shape_size - k * average_x)
                if to_print:
                    print("polygon_shape_size =", shape_size)
                    print("shape_x =", average_x)
                    print("shape_y =", average_y)
                    print("deviation =", deviation)
                #size_score = math.exp(-2 * deviation)
                size_score = 1 / (math.log(deviation + 1))
                #print("size_score: ", size_score)
                if size_score > 0.35:
                    size_score = (6//n_of_polygons) * size_score
                    if size_score > 4:
                        size_score = 4
                    #print(f"colour:{segment.colour} polygon positive size_score =", size_score)
                    total_score += size_score
                elif size_score < 0.35 and average_x < 90:
                    negative_score = 2 * (math.log(deviation + 1))
                    #print(f"colour:{segment.colour} polygon negative size_score =", -negative_score)
                    total_score -= negative_score
                elif size_score >=0 and average_x >100:
                    total_score +=1

        elif segment.segment_type == SegmentType.STAR:
            n_of_stars +=1
            #Score a star based on size (bigger preferred on outside of eye and smaller nearer the inner corner)
            x_values = segment.points_array[:, 0]
            y_values = segment.points_array[:, 1]
            average_x = np.mean(x_values)
            average_y = np.mean(y_values)
            shape_size = max((max(x_values)-min(x_values)) , (max(y_values)-min(y_values)))
            if average_x <= 100:
                k = 0.3  # Default for small x-values
            else:
                k = 0.32 #+ 0.002 * (average_x - 100)  # Gradual increase for large x-values
            if average_y < 70 and average_x < 110:
                k *= 0.7
            deviation = abs(shape_size - k * average_x)
            if to_print:
                print("star_shape_size =", shape_size)
                print("star_shape_x =", average_x)
                print("shape_y =", average_y)
                print("star_deviation =", deviation)
            #size_score = math.exp(-2 * deviation)
            size_score = 1 / (math.log(deviation + 1))
            #print("size_score: ",size_score)
            if size_score > 0.35:
                size_score = (6//n_of_stars) * size_score
                if size_score >4:
                    size_score = 4
                if to_print:
                    print(f"colour:{segment.colour} star_size_score =", size_score)
                total_score += size_score
            elif size_score < 0.3 and average_x<95:
                negative_score =2 * (math.log(deviation + 1))
                if to_print:
                    print(f"colour:{segment.colour} star_size_score =", -negative_score)
                total_score -= negative_score
            elif size_score >= 0 and average_x > 100:
                total_score += 1

    penalise_n_of_segments = len(segments) / 4
    if penalise_n_of_segments < 1:
        # So score doesn't get bigger with smaller n of segments
        penalise_n_of_segments = 1
    total_score = total_score / penalise_n_of_segments

    if n_of_stars + n_of_polygons > 4:
        total_score -=5
    else:
        if n_of_polygons >2:
            total_score -=2
        if n_of_stars >2:
            total_score -=2

    if n_of_jumps >4:
        total_score -=6

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


