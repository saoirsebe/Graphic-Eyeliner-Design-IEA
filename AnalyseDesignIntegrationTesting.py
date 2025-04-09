import pytest
from Segments import LineSegment, StarSegment
from EyelinerDesign import EyelinerDesign
from AnalyseDesign import analyse_negative
from A import SegmentType, StartMode, StarType, min_negative_score, upper_eyelid_coords


def create_design_case1():
    """
    Create design case 1:
    Uses a line segment as root (with start from upper eyelid) and a child segment.
    Expected: analyse_negative(design) returns a score < min_negative_score.
    """
    line = LineSegment(SegmentType.LINE, upper_eyelid_coords[-1], StartMode.JUMP,
                       30, 30, 1, 2, 'red', 0, True, 0.4, 0, 0)
    line1 = LineSegment(SegmentType.LINE, (80,120), StartMode.CONNECT_MID,
                        50, 150, 1, 2, 'blue', 0.7, False, 0.7, 0.7, 0)
    design = EyelinerDesign(line)
    line.add_child_segment(line1)
    design.render_design(show=False)
    return design

def create_design_case2():
    """
    Create design case 2:
    .
    """
    line = LineSegment(SegmentType.LINE, (100,150), StartMode.JUMP,
                       40, 40, 2, 2, 'green', 0, True, 0.5, 0, 0)
    line1 = LineSegment(SegmentType.LINE, (120,160), StartMode.CONNECT_MID,
                        60, 100, 2, 2, 'purple', 0.8, False, 0.8, 0.8, 0)
    design = EyelinerDesign(line)
    line.add_child_segment(line1)
    design.render_design(show=False)
    return design

def create_design_case3():
    """
    Create design case 3:

    """
    star = StarSegment(SegmentType.STAR, upper_eyelid_coords[-1], "red", StarType.STRAIGHT, 7,7,5,0.3,StartMode.CONNECT,1.5,260,True)
    line1 = LineSegment(SegmentType.LINE, (80,120), StartMode.CONNECT_MID, 50, 150, 1, 2, 'blue', 0.7, False, 0.7, 0.7, 0)
    design = EyelinerDesign(star)
    star.add_child_segment(line1)
    design.render_design(show=False)
    return design

def create_design_case4():
    """
    Create design case 4:

    """
    line = LineSegment(SegmentType.LINE, upper_eyelid_coords[-1], StartMode.JUMP, 30, 30, 1, 2, 'red', 0, True, 0.4, 0,0)
    line1 = LineSegment(SegmentType.LINE, (80, 120), StartMode.CONNECT_MID, 30, 150, 1, 2, 'blue', 0.7, False, 0.7, 0.7,0)
    line2 = LineSegment(SegmentType.LINE, (100, 130), StartMode.JUMP, 60, 330, 3, 1.5, 'green', 0.3, False, 0.5, 0, 0.3)

    design = EyelinerDesign(line)
    line.add_child_segment(line1)
    line1.add_child_segment(line2)
    design.render_design(show=False)
    return design


def create_design_case5():
    """
    Create design case 5:

    """
    star = StarSegment(SegmentType.STAR, upper_eyelid_coords[-1], "violet", StarType.STRAIGHT, 7, 7, 5, 0.3,StartMode.CONNECT, 1.5, 260, True)
    line = LineSegment(SegmentType.LINE, upper_eyelid_coords[-1], StartMode.JUMP, 20, 30, 1, 2, 'purple', 0, True, 0.4,0, 0)
    line1 = LineSegment(SegmentType.LINE, (80, 120), StartMode.CONNECT_MID, 30, 230, 1, 2, 'pink', 0.3, False, 0.5, 0.7,
                        0)

    design = EyelinerDesign(line)
    line.add_child_segment(star)
    star.add_child_segment(line1)
    design.render_design(show=False)
    return design

def create_design_case6():
    """
    Create design case 6:

    """
    star1 = StarSegment(SegmentType.STAR, (125,97), "violet", StarType.STRAIGHT, 7,7,5,0.3,StartMode.JUMP,1.5,260,True)
    line = LineSegment(SegmentType.LINE, upper_eyelid_coords[-1], StartMode.JUMP, 20, 30, 1, 2, 'purple', 0, True, 0.4,0, 0)
    line1 = LineSegment(SegmentType.LINE, (80, 120), StartMode.CONNECT_MID, 30, 230, 1, 2, 'pink', 0.3, False, 0.5, 0.7,
                        0)

    design = EyelinerDesign(line)
    line.add_child_segment(star1)
    star1.add_child_segment(line1)
    design.render_design(show=False)
    return design

@pytest.mark.parametrize("design, expected_relation", [
    # Invalid design: Slight eye overlap
    (create_design_case1(), "<"),
    # Invalid design: Segments too high
    (create_design_case2(), "<"),
    # Invalid design: Line segment inside star segment
    (create_design_case3(), "<"),
    # Invalid design: Too many overlaps
    (create_design_case4(), "<"),
    # Valid design
    (create_design_case5(), ">="),
    # Invalid design: Line segment inside star segment
    (create_design_case6(), "<"),
])
def test_analyse_negative(design, expected_relation):
    score = analyse_negative(design)
    if expected_relation == "<":
        assert score < min_negative_score, f"Expected score {score} to be less than {min_negative_score}"
    else:
        assert score >= min_negative_score, f"Expected score {score} to be >= {min_negative_score}"
