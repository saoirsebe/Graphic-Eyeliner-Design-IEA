from scipy.spatial import cKDTree
from A import SegmentType, StartMode, min_fitness_score
from AestheticAnalysis import analyse_design_shapes
import numpy as np
from EyelinerWingGeneration import get_quadratic_points

def check_overlaps(segment1,segment2, segment1_tree = None):
    overlaps = 0
    if segment1_tree is None:
        segment1_tree = cKDTree(segment1)  # Build KD-tree for the first set of points
    # Query all points in segment2 to find neighbors in segment1 within distance 0.075
    overlap_lists = segment1_tree.query_ball_point(segment2, 0.075)
    # Count the number of overlaps (non-empty lists indicate at least one neighbor)
    segment_overlaps = sum(len(neighbors) > 0 for neighbors in overlap_lists)

    if segment_overlaps:
        overlaps += int((segment_overlaps / (len(segment1) + len(segment2))) * 100)

    return overlaps

def check_design_overlaps(i, segments):
    segment = segments[i].points_array
    first_1 = int(len(segment) * 0.025)
    segment = segment[first_1:-first_1]
    segment_tree = cKDTree(segment)
    overlaps = 0
    for j in range(i + 1, len(segments) - 1):
        segment_j = segments[j].points_array
        #Take of first and las 2.5% of segments as they are allowed to meet at ends
        first_1 = int(len(segment_j) * 0.025)
        segment_j = segment_j[first_1:-first_1]

        #Take of extra 2.5% if they are meant to connect
        if j == (i + 1) and (segments[j].start_mode == StartMode.CONNECT_MID or segments[j].start_mode == StartMode.CONNECT):
            first_1 = int(len(segment_j) * 0.025)
            segment_j = segment_j[first_1:]
        elif j == (i + 1) and (segments[j].start_mode == StartMode.CONNECT or segments[j].start_mode == StartMode.SPLIT):
            first_1 = int(len(segment) * 0.025)
            segment = segment[:-first_1]

        overlaps += check_overlaps(segment, segment_j, segment_tree)

        if -overlaps < min_fitness_score: #Return if less than min_fitness_score to save processing time
            return overlaps

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

def wing_angle(node1,node2):
    if node1.segment_type == SegmentType.LINE and node2.segment_type == SegmentType.LINE:
        if node2.start_mode == StartMode.CONNECT and (0 < node1.absolute_angle < 70) and (142.5 < node2.relative_angle < 172.5 or 187.5 < node2.relative_angle < 217.5):
             return 3
    return 0

def analyse_negative(design):
    segments = design.get_all_nodes()
    score = 0  # Count how many overlaps there are in this gene
    # Compare each pair of segments for overlap
    for i in range(len(segments)-1):
        eye_overlaps = is_in_eye(segments[i])
        if eye_overlaps > 5:
            return min_fitness_score *2
        else:
            score-=eye_overlaps
        if score < min_fitness_score:
            return score
        score = score - check_design_overlaps(i, segments)
        if score < min_fitness_score:
            return score
    return score

def analyse_positive(design):
    segments = design.get_all_nodes()
    score=0
    #If starts with a wing angle
    for child in design.root.children:
        score += wing_angle(design.root, child)

    score += analyse_design_shapes(design)
    score = score * len(segments) * 0.5  # Higher score for designs with more segments
    return score