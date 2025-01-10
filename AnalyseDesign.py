from A import SegmentType, StartMode, min_fitness_score
from AestheticAnalysis import analyse_design_shapes
import numpy as np
from EyelinerWingGeneration import get_quadratic_points

def check_overlap(i, segments):
    segment = segments[i].points_array
    overlaps = 0
    for j in range(i + 1, len(segments) - 1):
        if segments[j].segment_type != SegmentType.BRANCH_POINT and segments[j].segment_type != SegmentType.END_POINT:
            overlap_found = False
            segment_overlaps = 0
            segment_j = segments[j].points_array

            if j == (i + 1) and (segments[j].start_mode == StartMode.CONNECT_MID or segments[j].start_mode == StartMode.CONNECT):
                first_1 = int(len(segment_j) * 0.05)
                segment_j = segment_j[first_1:]
            elif j == (i + 1) and (segments[j].start_mode == StartMode.CONNECT or segments[j].start_mode == StartMode.SPLIT):
                first_1 = int(len(segment) * 0.05)
                segment = segment[:-first_1]
            for point1 in segment:
                for point2 in segment_j:
                    # Calculate Euclidean distance between point1 and point2
                    distance = np.linalg.norm(point1 - point2)
                    # Check if distance is within the tolerance
                    if distance <= 0.075:
                        overlap_found = True
                        segment_overlaps += 1
            if overlap_found:
                overlaps += int((segment_overlaps / (len(segment) + len(segment_j))) * 100)

    return overlaps

def is_in_eye(segment):
    overlap = 0
    # Get eye shape boundary points
    upper_x, upper_y = get_quadratic_points(-0.5, 0, 1, -1, 1)
    lower_x, lower_y = get_quadratic_points(0.5, 0, 0, -1, 1)
    upper_x, upper_y = np.array(upper_x) * 3, np.array(upper_y) * 3
    lower_x, lower_y = np.array(lower_x) * 3, np.array(lower_y) * 3

    segment_array = segment.points_array
    if not isinstance(segment_array, np.ndarray):
        print("segment is NOT a NumPy array")
        print("Type:", segment.segment_type)
        print("Contents:", segment_array)

    upper_y_interp = np.interp(segment_array[:, 0], upper_x, upper_y)
    lower_y_interp = np.interp(segment_array[:, 0], lower_x, lower_y)
    inside = (lower_y_interp <= segment_array[:, 1]) & (segment_array[:, 1] <= upper_y_interp)
    overlap = np.sum(inside)
    score = int((overlap / len(segment_array)) * 100)
    if score > 0:
        return max(score, 1)
    else:
        return 0

def wing_angle(i, segments):
    if i + 1 < len(segments) and segments[i + 1].segment_type != SegmentType.END_POINT and segments[i].segment_type == SegmentType.LINE:
        next_int = i + 1
        while segments[next_int].segment_type == SegmentType.BRANCH_POINT:
            next_int +=1
        if segments[next_int].segment_type == SegmentType.LINE:
            if segments[next_int].start_mode == StartMode.CONNECT and (110 < segments[next_int].relative_angle < 160 or 200 < segments[next_int].relative_angle < 250):
                 return 5
    return 0

def analyse_negative(design):
    segments = design.segments
    score = 0  # Count how many overlaps there are in this gene
    # Compare each pair of segments for overlap
    for i in range(len(segments)-1):
        if segments[i].segment_type != SegmentType.BRANCH_POINT and segments[i].segment_type != SegmentType.END_POINT:
            score = score - check_overlap(i, segments)
            if score < min_fitness_score:
                return score
            score = score - is_in_eye(segments[i])
            if score < min_fitness_score:
                return score
    return score

def analyse_positive(design):
    segments = design.segments
    score = 0
    for i in range(len(segments)-1):
        score = score + wing_angle(i, segments)
        score +=1 #Higher score for designs with more segments
    score += analyse_design_shapes(design)
    return score