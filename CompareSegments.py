import copy

import numpy as np
from A import *
from AnalyseDesign import fix_overlaps_shape_overlaps
from EyelinerDesign import random_lines_corners_list
from Segments import LineSegment, generate_valid_start, random_segment_colour, create_segment
from IrregularPolygonSegment import *

def compare_segments(seg1, seg2):
    diff = 0

    if seg1.segment_type != seg2.segment_type:
        return 10

    #Compare colours:
    def compare_colour(c1, c2):
        if c1 == c2:
            return 0.0
        elif c2 in similar_colours.get(c1, []):
            return 0.5
        else:
            return 1.0

    #Compare segments based on type.
    if seg1.segment_type == SegmentType.LINE:
        # Numeric comparisons (normalized by range widths)
        diff += abs(seg1.length - seg2.length) / (length_range[1] - length_range[0])
        diff += abs(seg1.relative_angle - seg2.relative_angle) / (direction_range[1] - direction_range[0])
        diff += abs(seg1.start_thickness - seg2.start_thickness) / (thickness_range[1] - thickness_range[0])
        diff += abs(seg1.end_thickness - seg2.end_thickness) / (thickness_range[1] - thickness_range[0])
        diff += abs(seg1.curviness - seg2.curviness) / (curviness_range[1] - curviness_range[0])
        diff += abs(seg1.curve_location - seg2.curve_location) / (
                    relative_location_range[1] - relative_location_range[0])
        diff += abs(seg1.start_location - seg2.start_location) / (
                    relative_location_range[1] - relative_location_range[0])
        diff += abs(seg1.split_point - seg2.split_point) / (relative_location_range[1] - relative_location_range[0])

        # Categorical attributes
        if seg1.start_mode != seg2.start_mode:
            diff += 1.0
        diff += compare_colour(seg1.colour, seg2.colour)

    elif seg1.segment_type == SegmentType.STAR:
        # Numeric comparisons
        diff += abs(seg1.radius - seg2.radius) / (radius_range[1] - radius_range[0])
        diff += abs(seg1.arm_length - seg2.arm_length) / (arm_length_range[1] - arm_length_range[0])
        diff += abs(seg1.num_points - seg2.num_points) / (num_points_range[1] - num_points_range[0])
        diff += abs(seg1.asymmetry - seg2.asymmetry) / (asymmetry_range[1] - asymmetry_range[0])
        diff += abs(seg1.relative_angle - seg2.relative_angle) / (direction_range[1] - direction_range[0])
        diff += abs(seg1.end_thickness - seg2.end_thickness) / (thickness_range[1] - thickness_range[0])

        # Categorical attributes
        if seg1.start_mode != seg2.start_mode:
            diff += 1.0
        if seg1.star_type != seg2.star_type:
            diff += 1.0
        diff += compare_colour(seg1.colour, seg2.colour)

        # Boolean for fill
        if seg1.fill != seg2.fill:
            diff += 0.5

    elif seg1.segment_type == SegmentType.IRREGULAR_POLYGON:
        # Numeric comparisons
        diff += abs(seg1.end_thickness - seg2.end_thickness) / (thickness_range[1] - thickness_range[0])
        diff += abs(seg1.relative_angle - seg2.relative_angle) / (direction_range[1] - direction_range[0])

        # Categorical
        if seg1.start_mode != seg2.start_mode:
            diff += 1.0
        diff += compare_colour(seg1.colour, seg2.colour)

        # Compare bounding_size, assumed to be a tuple (width, height)
        diff += abs(seg1.bounding_size[0] - seg2.bounding_size[0]) / (
                    random_shape_size_range[1] - random_shape_size_range[0])
        diff += abs(seg1.bounding_size[1] - seg2.bounding_size[1]) / (
                    random_shape_size_range[1] - random_shape_size_range[0])

        # Compare corners, assumed to be a list of (x, y) points.
        if len(seg1.corners) == len(seg2.corners):
            for pt1, pt2 in zip(seg1.corners, seg2.corners):
                # Normalize by a characteristic size (here, using the width of random_shape_size_range)
                diff += np.linalg.norm(np.array(pt1) - np.array(pt2)) / (
                            random_shape_size_range[1] - random_shape_size_range[0])
        else:
            diff += 1.0  # Penalty if the number of corners differs

        #Compare lines_list (list of edge segments with attributes curviness and curve_location)
        if len(seg1.lines_list) == len(seg2.lines_list):
            for edge1, edge2 in zip(seg1.lines_list, seg2.lines_list):
                diff += abs(edge1.curviness - edge2.curviness) / (curviness_range[1] - curviness_range[0])
                diff += abs(edge1.curve_location - edge2.curve_location) / (
                            relative_location_range[1] - relative_location_range[0])
        else:
            diff += 1.0

        #Compare fill
        if seg1.fill != seg2.fill:
            diff += 0.5

        #Compare is_eyeliner_wing if it exists:
        if hasattr(seg1, 'is_eyeliner_wing') and hasattr(seg2, 'is_eyeliner_wing'):
            if seg1.is_eyeliner_wing != seg2.is_eyeliner_wing:
                diff += 0.5

    return diff

def random_irregular_polygon(ax=None):

    segment_start = generate_valid_start()
    random_colour = random_segment_colour("blue")
    new_shape_overlaps = max_shape_overlaps + 21
    while new_shape_overlaps > 20:
        n_of_corners = round(random_normal_within_range(5, 1, num_points_range))
        corners, lines = random_lines_corners_list(n_of_corners)
        bounding_size_x = random_normal_within_range(50, 30, random_shape_size_range)
        new_segment = create_segment(
            segment_type=SegmentType.IRREGULAR_POLYGON,
            start=segment_start,
            start_mode=StartMode.CONNECT if random.random() < 0.3 else StartMode.JUMP,
            end_thickness=random_normal_within_range(2, 2, thickness_range),
            relative_angle=random.uniform(*direction_range),
            colour=random_colour,
            bounding_size=(bounding_size_x, random_normal_within_range(bounding_size_x, 30, random_shape_size_range)),
            corners=corners,
            lines_list=lines,
            fill=random.choice([True, False]),
        )
        prev_array = np.array([new_segment.start])
        prev_angle = 0
        prev_colour = new_segment.colour
        prev_end_thickness = new_segment.end_thickness
        new_segment.render(prev_array, prev_angle, prev_colour, prev_end_thickness)

        new_shape_overlaps = fix_overlaps_shape_overlaps(new_segment, new_segment.lines_list, ax)


    return new_segment
"""
#line = LineSegment(SegmentType.LINE, (100,10), StartMode.CONNECT_MID, 100, 0, 3, 2, 'lightblue', 0, True, 0.5, 0, 0)
#line2 = LineSegment(SegmentType.LINE, (50,100), StartMode.JUMP, 30, 0, 1, 2, 'blue', 0, True, 0.5, 0, 0)

fig, ax_n = plt.subplots(figsize=(3, 3))
polygon1 = random_irregular_polygon(ax_n)
prev_array = np.array([polygon1.start])
prev_angle = 0
prev_colour = polygon1.colour
prev_end_thickness= polygon1.end_thickness
polygon1.render(prev_array, prev_angle, prev_colour, prev_end_thickness, ax_n=ax_n)
polygon2 = copy.deepcopy(polygon1)
polygon2.mutate()
polygon2.mutate()
polygon2.mutate()
polygon2.start = (polygon2.start[0] + 100, polygon2.start[1])

polygon2.render(prev_array, prev_angle, prev_colour, prev_end_thickness, ax_n=ax_n)

fig.show()

difference = compare_segments(polygon1,polygon2)
print("difference =",difference)

"""