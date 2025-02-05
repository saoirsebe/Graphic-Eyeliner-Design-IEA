from scipy.spatial import cKDTree
from A import SegmentType, StartMode, min_fitness_score, max_shape_overlaps, upper_eyelid_x, lower_eyelid_x, \
    upper_eyelid_y, lower_eyelid_y, SegmentType
from AestheticAnalysis import analyse_design_shapes
import numpy as np

from ParentSegment import point_in_array


def remove_ends_of_line(line1_array, line2_array):
    first_2 = int(len(line1_array) * 0.02)
    line1_array = line1_array[:-first_2]
    first_2 = int(len(line2_array) * 0.02)
    line2_array = line2_array[first_2:]
    return line1_array, line2_array

def check_overlaps(segment2_array, segment1_tree, tolerance=0.025):
    overlapping_indices = set()

    for p in segment2_array:
        indices = segment1_tree.query_ball_point(p, tolerance)
        overlapping_indices.update(indices)  # Add all indices found within the tolerance

    overlaps = len(overlapping_indices)

    return overlaps

def check_shape_edge_overlaps(segment1_array, segment2_array):
    segment1_tree = cKDTree(segment1_array)  # Build KD-tree for the first set of points
    overlaps = check_overlaps(segment2_array, segment1_tree)
    return overlaps

def check_segment_overlaps(segment1, segment2, segment1_tree = None):
    segment2_array = segment2.points_array
    if segment2.segment_type == SegmentType.LINE:
        first_25 = int(len(segment2_array) * 0.025)
        segment2_array = segment2_array[first_25:-first_25]
        if segment2.start_mode == StartMode.SPLIT:
            #Remove the split point from array as it will overlap with the previous segment:
            split_point_point_index = point_in_array(segment2.points_array, segment2.split_point)
            segment2_array = np.delete(segment2_array, split_point_point_index, axis=0)

    if segment1_tree is None:
        segment1_array = segment1.points_array
        if segment1.segment_type == SegmentType.LINE:
            first_25 = int(len(segment1_array) * 0.025)
            segment1_array = segment1_array[first_25:-first_25]
        segment1_tree = cKDTree(segment1_array)  # Build KD-tree for the first set of points

    overlaps = check_overlaps(segment2_array, segment1_tree)
    #if n_of_overlaps > 0:
    #    total_points = len(segment1_tree.data) + len(segment2_array)
    #    overlaps = int((n_of_overlaps / total_points) * 100)  # Calculate percentage overlap
    #else:
    #    overlaps = 0

    return overlaps

def check_design_overlaps(i, segments):
    segment = segments[i]
    segment_array = segment.points_array
    if len(segments)==0:
        print("len(segments)==0")
    if len(segment_array) <50:
        print(f"segment {segment.segment_type} if length", len(segment_array))
    if segment.segment_type == SegmentType.LINE:
        first_25 = int(len(segment_array) * 0.025)
        segment_array = segment_array[first_25:-first_25]
    segment_tree = cKDTree(segment_array)
    overlaps = 0
    for j in range(i + 1, len(segments)):
        segment_j = segments[j]
        overlaps += check_segment_overlaps(segment, segment_j, segment_tree)
        if overlaps > 0:
            print("segment.segment_type:",segment.segment_type)
            print("segment_j.segment_type:",segment_j.segment_type)
            print("overlaps:", overlaps)
        if -overlaps < min_fitness_score: #Return if less than min_fitness_score to save processing time
            return overlaps

    return overlaps

def percentage_is_in_eye(segment):
    segment_array = segment.points_array
    #Remove ends so segment can touch eye but not go in
    if segment.segment_type == SegmentType.LINE:
        first_2 = int(len(segment_array) * 0.02)
        segment_array = segment_array[first_2:-first_2]

    if not isinstance(segment_array, np.ndarray):
        print("segment is NOT a NumPy array")
        print("Contents:", segment_array)

    if segment_array.ndim == 1:
        print("Segment Array:", segment_array)
        print("Shape:", segment_array.shape)

    if len(segment_array) == 0:
        print("Segment Array is empty")
        print("len(segment_array)", len(segment_array))
        print("first2:",first_2)
        print("Segment", segment)

    tolerance = 0.01
    upper_y_interp = np.interp(segment_array[:, 0], upper_eyelid_x, upper_eyelid_y) - tolerance
    lower_y_interp = np.interp(segment_array[:, 0], lower_eyelid_x, lower_eyelid_y) + tolerance
    inside = (lower_y_interp <= segment_array[:, 1]) & (segment_array[:, 1] <= upper_y_interp)
    overlap = np.sum(inside)
    #score = int((overlap / len(segment_array)) * 100)
    score = overlap
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
    print("design:")
    segments = design.get_all_nodes()
    score = 0  # Count how many overlaps there are in this gene
    # Compare each pair of segments for overlap
    for i in range(len(segments)):
        eye_overlaps = percentage_is_in_eye(segments[i])
        if eye_overlaps>0:
            print("percentage_in_eye=", eye_overlaps)
        if eye_overlaps > 5:
            return min_fitness_score *2
        else:
            score-=eye_overlaps
        if score < min_fitness_score:
            return score
        if i!=len(segments)-1:
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
    score = score + (len(segments) * 0.5)  # Higher score for designs with more segments
    return score


def shape_overlaps(lines):
    sorted_lines = sorted(lines, key=lambda line: line.curviness, reverse=True)
    #Tries to fix overlaps by making lines less curvey:
    for i, line in enumerate(sorted_lines):
        try_again = True
        try_again_count = 0
        while try_again and try_again_count < 8:
            line_overlaps = 0
            line_array = line.points_array
            for j in range(0, len(lines)):
                if i!=j:
                    line_j_array = lines[j].points_array
                    if j == (i + 1):
                        remove_ends_of_line(line_array, line_j_array)
                    line_overlaps += check_shape_edge_overlaps(line_array, line_j_array)
            try_again = False

            if line_overlaps>max_shape_overlaps:
                if line.curviness >0.025:
                    line.curviness -=0.025
                    try_again = True

            try_again_count += 1

    overlaps = 0
    #Check final overlaps:
    for i, line in enumerate(sorted_lines):
        line_overlaps = 0
        line_array = line.points_array
        for j in range(i + 1, len(lines)):
            line_j_array = lines[j].points_array
            if j == (i + 1):
                remove_ends_of_line(line_array, line_j_array)
            line_overlaps += check_shape_edge_overlaps(line_array, line_j_array)
            if line_overlaps > max_shape_overlaps:  # Return to save processing time
                return overlaps

        overlaps += line_overlaps
        if overlaps > max_shape_overlaps:  # Return to save processing time
            return overlaps

    return overlaps