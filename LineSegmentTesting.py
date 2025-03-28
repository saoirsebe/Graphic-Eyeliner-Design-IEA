import pytest

from A import *
from Segments import LineSegment


def test_line_mutation_attributes_within_range():
    line = LineSegment(SegmentType.LINE, (50,100), StartMode.JUMP, 70, 0, 2, 1, 'red', 0.7, True, 0.4, 0, 0)
    for count in range(10):
        line.mutate()
        assert line.segment_type == SegmentType.LINE
        assert isinstance(line.start_mode, StartMode)
        assert start_x_range[0] <= line.start[0] <= start_x_range[1]
        assert start_y_range[0] <= line.start[1] <= start_y_range[1]
        assert relative_location_range[0] <= line.start_location <= relative_location_range[1]
        assert relative_location_range[0] <= line.split_point <= relative_location_range[1]
        assert thickness_range[0] <= line.end_thickness <= thickness_range[1]
        assert direction_range[0] <= line.relative_angle <= direction_range[1]
        assert line.colour in colour_options
        assert length_range[0] <= line.length <= length_range[1]
        assert thickness_range[0] <= line.start_thickness <= thickness_range[1]
        assert curviness_range[0] <= line.curviness <= curviness_range[1]
        assert isinstance(line.curve_left, bool)
        assert relative_location_range[0] <= line.curve_location <= relative_location_range[1]

    print("Mutation attributes range tests passed successfully!")

