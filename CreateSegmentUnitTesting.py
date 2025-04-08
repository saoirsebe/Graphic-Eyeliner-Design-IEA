import pytest
from RandomGene import create_segment
from A import SegmentType, StarType, StartMode

@pytest.mark.parametrize("segment_type, start_mode, input_kwargs, expected_attrs", [
    # Test for a LineSegment with start_mode CONNECT.
    (
        SegmentType.LINE,
        StartMode.CONNECT,
        {
            "start": (0, 0),
            "end_thickness": 1.5,
            "relative_angle": 45,
            "colour": "red",
            # Even if a start_location is provided, for CONNECT it is forced to 1.
            "start_thickness": 3,
            "length": 10,
            "curviness": 0.5,
            "curve_left": False,  # used to set curve_left
            "curve_location": 0.7,
            "start_location": 0.8,  # provided value, but should be ignored
            "split_location": 0.6,     # provided value, but should be overridden to 0
        },
        {
            "segment_type": SegmentType.LINE,
            "start": (0, 0),
            "end_thickness": 1.5,
            "relative_angle": 45,
            "colour": "red",
            "start_thickness": 3,
            "length": 10,
            "curviness": 0.5,
            "curve_left": False,    # because curve_90 is False
            "curve_location": 0.7,
            # For a CONNECT mode (and not CONNECT_MID), start_location is forced to 1.
            "start_location": 1,
            # And split_location is forced to 0.
            "split_location": 0,
        }
    ),
    # Test for a LineSegment with start_mode CONNECT_MID.
    (
        SegmentType.LINE,
        StartMode.CONNECT_MID,
        {
            "start": (0, 0),
            "end_thickness": 1.5,
            "relative_angle": 60,
            "colour": "orange",
            "start_thickness": 2,
            "length": 8,
            "curviness": 0.2,
            "curve_left": True,   # so curve_left True
            "curve_location": 0.65,
            "start_location": 0.9,  # in CONNECT_MID, this value is used
            "split_location": 0.3,     # for CONNECT_MID, split_location is forced to 0
        },
        {
            "segment_type": SegmentType.LINE,
            "start": (0, 0),
            "end_thickness": 1.5,
            "relative_angle": 60,
            "colour": "orange",
            "start_thickness": 2,
            "length": 8,
            "curviness": 0.2,
            "curve_left": True,
            "curve_location": 0.65,
            # For CONNECT_MID, start_location is preserved.
            "start_location": 0.9,
            # And split_location is forced to 0.
            "split_location": 0,
        }
    ),
    # Test for a LineSegment with start_mode SPLIT.
    (
        SegmentType.LINE,
        StartMode.SPLIT,
        {
            "start": (0, 0),
            "end_thickness": 2,
            "relative_angle": 30,
            "colour": "purple",
            "start_thickness": 4,
            "length": 15,
            "curviness": 0.4,
            "curve_left": True,   # so curve_left True
            "curve_location": 0.8,
            "start_location": 0.5,  # provided value ignored for SPLIT
            "split_location": 0.25,    # this value is used in SPLIT mode
        },
        {
            "segment_type": SegmentType.LINE,
            "start": (0, 0),
            "end_thickness": 2,
            "relative_angle": 30,
            "colour": "purple",
            "start_thickness": 4,
            "length": 15,
            "curviness": 0.4,
            "curve_left": True,
            "curve_location": 0.8,
            # For SPLIT mode, start_location is forced to 1.
            "start_location": 1,
            # And split_location is taken from the kwargs.
            "split_location": 0.25,
        }
    ),
    # Test for a StarSegment.
    (
        SegmentType.STAR,
        StartMode.JUMP,
        {
            "start": (1, 1),
            "end_thickness": 2,
            "relative_angle": 90,
            "colour": "green",
            "radius": 0.9,
            "arm_length": 1.2,
            "num_points": 7,
            "asymmetry": 0.2,
            "star_type": StarType.CURVED,
            "fill": True,
        },
        {
            "segment_type": SegmentType.STAR,
            "start": (1, 1),
            "end_thickness": 2,
            "relative_angle": 90,
            "colour": "green",
            "radius": 0.9,
            "arm_length": 1.2,
            "num_points": 7,
            "asymmetry": 0.2,
            "star_type": StarType.CURVED,
            "fill": True,
        }
    ),
    # Test for an IrregularPolygonSegment.
    (
        SegmentType.IRREGULAR_POLYGON,
        StartMode.SPLIT,
        {
            "start": (2, 2),
            "end_thickness": 3,
            "relative_angle": 135,
            "colour": "black",
            "bounding_size": (5, 5),
            "corners": [(0, 0), (1, 0), (1, 1), (0, 1)],
            "lines_list": ["line1", "line2"],
            "fill": True,
            "is_eyeliner_wing": True,
        },
        {
            "segment_type": SegmentType.IRREGULAR_POLYGON,
            "start": (2, 2),
            "end_thickness": 3,
            "relative_angle": 135,
            "colour": "black",
            "bounding_size": (5, 5),
            "corners": [(0, 0), (1, 0), (1, 1), (0, 1)],
            "lines_list": ["line1", "line2"],
            "fill": True,
            "is_eyeliner_wing": True,
        }
    )
])
def test_create_segment(segment_type, start_mode, input_kwargs, expected_attrs):
    # Do NOT insert start_mode into input_kwargs since it's already provided as a positional argument.
    # Extract required positional arguments.
    start = input_kwargs.pop("start")
    end_thickness = input_kwargs.pop("end_thickness")
    relative_angle = input_kwargs.pop("relative_angle")
    colour = input_kwargs.pop("colour")
    # Call the create_segment function.
    segment = create_segment(start, start_mode, segment_type, end_thickness, relative_angle, colour, **input_kwargs)
    # Check each expected attribute.
    for attr, expected in expected_attrs.items():
        actual = getattr(segment, attr, None)
        assert actual == expected, (
            f"For segment type {segment_type} with start_mode {start_mode}, attribute '{attr}': expected {expected}, got {actual}"
        )
