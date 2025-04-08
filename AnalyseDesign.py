from itertools import chain

from scipy.spatial import cKDTree
from seaborn import scatterplot
from A import SegmentType, StartMode, min_negative_score, max_shape_overlaps, upper_eyelid_x, lower_eyelid_x, \
    upper_eyelid_y, lower_eyelid_y, SegmentType, face_end_x_values, face_end_y_values
from AestheticAnalysis import analyse_design_shapes
import numpy as np
from ParentSegment import point_in_array


def remove_ends_of_line(line1_array, line2_array):
    first_2 = int(len(line1_array) * 0.15)
    line1_array = line1_array[:-first_2]
    first_2 = int(len(line2_array) * 0.15)
    line2_array = line2_array[first_2:]
    return line1_array, line2_array

"""
def check_overlaps(segment2_array, segment1_tree, tolerance=0.4):
    overlapping_indices = set()

    for p in segment2_array:
        indices = segment1_tree.query_ball_point(p, tolerance)
        overlapping_indices.update(indices)  # Add all indices found within the tolerance

    overlaps = len(overlapping_indices)

    return overlaps , overlapping_indices
"""
def check_overlaps(segment2_array, segment1_tree, tolerance=0.4):
    indices_lists = segment1_tree.query_ball_point(segment2_array, tolerance)
    overlapping_indices = set(chain.from_iterable(indices_lists))
    return len(overlapping_indices), overlapping_indices

def check_shape_edge_overlaps(segment1_array, segment2_array):
    segment1_tree = cKDTree(segment1_array)  # Build KD-tree for the first set of points
    overlaps, indices_of_line_overlaps = check_overlaps(segment2_array, segment1_tree, 0.2)
    return overlaps , indices_of_line_overlaps

def remove_both_ends(the_segment_array):
    first_25 = int(len(the_segment_array) * 0.025)
    return the_segment_array[first_25:-first_25]


import numpy as np


def any_points_inside_filled_polygon(points, polygon):
    """
    Efficiently checks if any points exist inside a pre-filled polygon (array of points).

    :param points: np.array of shape (N, 2) containing N points to check.
    :param polygon: np.array of shape (M, 2) containing all M points that make up the filled polygon.
    :return: True if at least one point in `points` exists inside `polygon`, False otherwise.
    """
    # Convert to sets for fast lookup
    polygon_set = set(map(tuple, polygon))
    points_set = set(map(tuple, points))

    # Check for any intersection
    return len(points_set & polygon_set) > 0

def check_new_segments_negative_score(design, new_segment):
    score = 0
    if is_outside_face_area(new_segment):
        return min_negative_score * 2
    eye_overlaps = is_in_eye(new_segment)
    if eye_overlaps > 0:
        return min_negative_score * 2

    average_x = np.mean(new_segment.points_array[:, 0])
    if average_x < 10:
        score -= 0.5

    average_y = np.mean(new_segment.points_array[:, 1])
    if average_y > 190:
        return min_negative_score * 2
    elif average_y > 150:
        score -= 1.5
    elif average_y < 50:
        score -= 0.5

    current_segments = design.get_all_nodes()
    for segment in current_segments:
        score -= check_segment_overlaps(segment,new_segment)
        if score < min_negative_score:
            return score

    return score

def check_segment_overlaps(segment1, segment2, segment1_tree = None):
    segment2_array = segment2.points_array
    if segment2.segment_type == SegmentType.LINE:
        if segment2.start_mode == StartMode.SPLIT:
            #Remove the split point from array as it will overlap with the previous segment:
            split_location_point_index = point_in_array(segment2.points_array, segment2.split_location)
            segment2_array = np.delete(segment2_array, split_location_point_index, axis=0)
        segment2_array = remove_both_ends(segment2_array)

    segment1_array = segment1.points_array
    if segment1_tree is None:
        if segment1.segment_type == SegmentType.LINE:
            segment1_array = remove_both_ends(segment1_array)
        segment1_tree = cKDTree(segment1_array)  # Build KD-tree for the first set of points

    overlaps , overlap_indices = check_overlaps(segment2_array, segment1_tree)
    #if n_of_overlaps > 0:
    #    total_points = len(segment1_tree.data) + len(segment2_array)
    #    overlaps = int((n_of_overlaps / total_points) * 100)  # Calculate percentage overlap
    #else:
    #    overlaps = 0
    inside_shape = False
    if segment1.segment_type == SegmentType.STAR or segment1.segment_type == SegmentType.IRREGULAR_POLYGON:
        inside_shape = any_points_inside_filled_polygon(segment2_array, segment1_array)
    elif segment2.segment_type == SegmentType.STAR or segment2.segment_type == SegmentType.IRREGULAR_POLYGON:
        inside_shape = any_points_inside_filled_polygon(segment1_array,segment2_array)
    if inside_shape:
        return -min_negative_score*2

    return overlaps

def check_design_overlaps(i, segments):
    segment = segments[i]
    segment_array = segment.points_array
    len_segments = len(segments)
    len_segment_array = len(segment_array)
    if len_segments==0:
        print("len(segments)==0")
    if len_segment_array <50:
        print(f"segment {segment.segment_type} is length", len_segment_array)
        print("segment_array:", segment_array)
    if segment.segment_type == SegmentType.LINE:
        first_2 = int(len_segment_array * 0.02)
        segment_array = segment_array[first_2:-first_2]
    segment_tree = cKDTree(segment_array)
    overlaps = 0
    for j in range(i + 1, len_segments):
        segment_j = segments[j]
        overlaps += check_segment_overlaps(segment, segment_j, segment_tree)
        #if overlaps > 0:
            #print("segment.segment_type:",segment.segment_type)
            #print("segment_j.segment_type:",segment_j.segment_type)
            #print("overlaps:", overlaps)
        if -overlaps < min_negative_score: #Return if less than min_negative_score to save processing time
            return overlaps

    return overlaps

def is_in_eye(segment):
    segment_array = segment.points_array
    len_segment_array = len(segment_array)
    #Remove ends so segment can touch eye but not go in
    if segment.segment_type == SegmentType.LINE:
        first_2 = int(len_segment_array * 0.02)
        segment_array = segment_array[first_2:-first_2]

    if not isinstance(segment_array, np.ndarray):
        print("segment is NOT a NumPy array")
        print("Contents:", segment_array)

    if segment_array.ndim == 1:
        print("Segment Array:", segment_array)
        print("Shape:", segment_array.shape)

    if len(segment_array) == 0:
        print("Segment Array is empty")
        print("len(segment_array)", len_segment_array)
        print("first2:", first_2)
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

def is_outside_face_area(segment):
    segment_array = segment.points_array
    if segment.segment_type == SegmentType.LINE:
        first_2 = int(len(segment_array) * 0.02)
        segment_array = segment_array[first_2:-first_2]
    x1 = face_end_x_values[0]
    x2 = face_end_x_values[-1]
    y1 = face_end_y_values[0]
    y2 = face_end_y_values[-1]

    for point in segment_array:
        px, py = point

        # Calculate cross product: (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        cross_product = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

        # If cross product is negative, the point is to the right
        if cross_product < 0:
            return True  # Return immediately if any point is found to the right

    return False  # No point to the right

def wing_angle(node1,node2):
    if node1.segment_type == SegmentType.LINE and node2.segment_type == SegmentType.LINE:
        if node2.start_mode == StartMode.CONNECT and (0 < node1.absolute_angle < 70) and (142.5 < node2.relative_angle < 172.5 or 187.5 < node2.relative_angle < 217.5):
             return 3
    return 0

def analyse_negative(design, to_print = False):
    #print("design:")
    segments = design.get_all_nodes()
    segments_outside_good = 0
    score = 0  # Count how many overlaps there are in this gene
    # Compare each pair of segments for overlap
    len_segments = len(segments)
    #print("N of segments = ",len_segments)
    for i in range(len_segments):
        average_x = np.mean(segments[i].points_array[:, 0])
        if average_x < 15:
            segments_outside_good+=1
            if to_print:
                print("average_x < 15")
            score -= 0.5

        average_y = np.mean(segments[i].points_array[:, 1])
        if average_y > 140:
            segments_outside_good += 1
            if to_print:
                print("average_y > 140")
            score -= 1.5
        elif average_y < 50:
            segments_outside_good += 1
            if to_print:
                print("average_y < 50")
            score -= 0.5

        eye_overlaps = is_in_eye(segments[i])
        if eye_overlaps>0:
            if to_print:
                print("eye_overlaps:", eye_overlaps)
            return min_negative_score *2
        #else:
        #    score-=eye_overlaps
        if is_outside_face_area(segments[i]):
            if to_print:
                print("is_outside_face_area")
            return min_negative_score * 2

        if score < min_negative_score:
            return score

        if i!=len_segments-1:
            segment_score = check_design_overlaps(i, segments)
            score  -= segment_score
            if to_print and segment_score>0:
                print(f"for {segments[i].colour} {segments[i].segment_type}, overlaps:", segment_score)
        if score < min_negative_score:
            return score

        if segments_outside_good >2:
            return min_negative_score * 2
    return score

def analyse_positive(design, to_print = False):
    #segments = design.get_all_nodes()
    score=0
    #If starts with a wing angle
    #for child in design.root.children:
    #    score += wing_angle(design.root, child)

    score += analyse_design_shapes(design, to_print)
    #score = score + (len(segments) * 0.25)  # Higher score for designs with more segments
    return score


def fix_overlaps_shape_overlaps(shape, lines, ax=None):
    """
    Aims to fix shape overlaps between edges by reducing curviness of lines (in order of curviest to least curvey lines) and moving curve location away from overlap location
    :param lines:
    :return:
    """
    # Store original index with each line before sorting
    indexed_lines = [(index, line) for index, line in enumerate(lines)]
    # Sort based on line.curviness
    sorted_lines_with_indices = sorted(indexed_lines, key=lambda item: item[1].curviness, reverse=True)
    len_lines = len(lines)

    #For rendering:
    prev_array = np.array([shape.start])
    prev_angle = 0
    prev_colour = shape.colour
    prev_end_thickness = shape.end_thickness
    #Tries to fix overlaps by making lines less curvey:
    for i in range(len_lines):
        try_again = True
        try_again_count = 0
        while try_again and try_again_count < 16:
            n_of_line_overlaps = 0
            indices_of_line_overlaps = []
            original_line_index = sorted_lines_with_indices[i][0]
            line_i_array = lines[original_line_index].points_array
            for j in range(len_lines):
                line_j_array = lines[j].points_array
                if original_line_index!=j and not np.array_equal(line_i_array, line_j_array):

                    if j == (original_line_index + 1)%len_lines:
                        line_i_array, line_j_array = remove_ends_of_line(line_i_array, line_j_array)
                    if j == (original_line_index - 1)%len_lines:
                        line_j_array, line_i_array = remove_ends_of_line(line_j_array, line_i_array)
                    current_n_of_line_overlaps, set_indices_of_line_overlaps = check_shape_edge_overlaps(line_i_array, line_j_array)
                    n_of_line_overlaps += current_n_of_line_overlaps
                    indices_of_line_overlaps.extend(list(set_indices_of_line_overlaps))

            try_again = False

            if n_of_line_overlaps>0:
                if lines[original_line_index].curviness >=0.025:
                    lines[original_line_index].curviness -=0.025
                    try_again = True
                if len(indices_of_line_overlaps)>0:

                    sorted_indices = sorted(indices_of_line_overlaps)
                    #If all indices are close together (only one overlap) then move curve location away from overlap point:
                    if max(sorted_indices) - min(sorted_indices) <= 10:
                        len_line_array = len(line_i_array)
                        half_index = len_line_array * 0.5
                        if lines[original_line_index].curve_location >0.5 and min(sorted_indices)>half_index:
                            lines[original_line_index].curve_location -= 0.025
                            try_again = True
                        elif lines[original_line_index].curve_location <0.5 and max(sorted_indices)<half_index:
                            lines[original_line_index].curve_location += 0.025
                            try_again = True

            try_again_count += 1
            if try_again:
                shape.render(prev_array, prev_angle, prev_colour,prev_end_thickness)  # Need to render to update points.array

    #if n_of_line_overlaps>0:
    #    return n_of_line_overlaps
    overlaps = 0
    #Check final overlaps:
    final_indices_of_line_overlaps = []
    for i in range(len_lines):
        line_array = lines[i].points_array
        for j in range(i + 1, len(lines)):
            line_j_array = lines[j].points_array
            if j == (i + 1) % len_lines:
                line_array, line_j_array = remove_ends_of_line(line_array, line_j_array)
            if j == (i - 1) % len_lines:
                line_j_array, line_array = remove_ends_of_line(line_j_array, line_array)
            n_overlaps, overlap_indices = check_shape_edge_overlaps(line_array, line_j_array)
            final_indices_of_line_overlaps.extend(list(overlap_indices))
            overlaps += n_overlaps

            if overlaps > 0:  # Return to save processing time
                return overlaps


    return overlaps

