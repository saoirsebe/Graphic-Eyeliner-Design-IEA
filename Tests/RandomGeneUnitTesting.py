import pytest
import random
import numpy as np
from RandomGene import random_gene_node, random_gene
from Segments import LineSegment


# --- Dummy Segment Class (used by our fakes) ---
class DummySegment:
    def __init__(self):
        self.colour = "blue"
        self.children = []
        self.start = (0, 0)
        self.absolute_angle = 0
        self.end_thickness = [1]
        # Set a default segment_type (adjust as needed)
        self.segment_type = "LINE"
        # We'll assume that when rendered, a points_array is set.
        self.points_array = np.array([[0, 0]])
    def render(self, points_array, absolute_angle, colour, end_thickness):
        # Simulate rendering by setting a nonempty points_array.
        self.points_array = np.array([[0, 0], [1, 1]])

# --- Fake Helper Functions for Deterministic Behavior ---

def fake_random_normal_within_range(mean, std_dev, value_range):
    # Always return the provided mean.
    return mean

def fake_random_segment_colour(prev_colour):
    return "blue"

def fake_set_prev_end_thickness_array(parent):
    return parent.end_thickness

def fake_set_prev_array(parent):
    return np.array([parent.start])

def fake_check_new_segments_negative_score(design, node):
    # For testing, return a fixed score high enough so no regeneration occurs.
    return 20

def fake_n_of_children_decreasing_likelihood(segment_number, depth, max_segments, base_mean, std_dev, value_range):
    # Force no children (stop recursion).
    return 0

# --- For random_gene ---
def fake_is_outside_face_area(root_node):
    # Force the branch that sets root_score = 2*min_negative_score.
    return True

def fake_is_in_eye(root_node):
    return 0

def fake_analyse_negative(design):
    # Return exactly min_negative_score so that analysis passes.
    return fake_min_negative_score

def fake_make_eyeliner_wing(random_colour):
    # Return a dummy segment representing an eyeliner wing.
    seg = DummySegment()
    seg.segment_type = "LINE"  # or set to a specific type if needed
    return seg

# Fake EyelinerDesign: a minimal design wrapper that provides get_all_nodes.
class FakeEyelinerDesign:
    def __init__(self, segment):
        self.root = segment
    def render_design(self, show=False):
        if hasattr(self.root, "render"):
            self.root.render(np.array([self.root.start]), self.root.absolute_angle, self.root.colour, self.root.end_thickness)
    def get_all_nodes(self):
        return [self.root]

def fake_EyelinerDesign(segment):
    return FakeEyelinerDesign(segment)

def fake_random_normal_within_range_for_gene(mean, std_dev, value_range):
    return mean

# --- Dummy Global Values for Testing ---
fake_min_negative_score = 10
fake_node_re_gen_max = 3
fake_max_segments = 100
fake_number_of_children_range = (0, 3)
fake_initial_gene_pool_size = 9

# --- Test for random_gene_node ---
def test_random_gene_node(monkeypatch):
    import RandomGene

    # Patch helper functions and globals in RandomGene.
    monkeypatch.setattr(RandomGene, "random_segment", lambda prev_colour=None, segment_start=None: LineSegment(
        segment_type=RandomGene.SegmentType.LINE,
        start=(0, 0),
        colour="blue",
        start_mode=RandomGene.StartMode.CONNECT,
        end_thickness=1,
        relative_angle=135,
    ))
    monkeypatch.setattr(RandomGene, "set_prev_end_thickness_array", fake_set_prev_end_thickness_array)
    monkeypatch.setattr(RandomGene, "set_prev_array", fake_set_prev_array)
    monkeypatch.setattr(RandomGene, "check_new_segments_negative_score", fake_check_new_segments_negative_score)
    monkeypatch.setattr(RandomGene, "n_of_children_decreasing_likelihood", fake_n_of_children_decreasing_likelihood)
    monkeypatch.setattr(RandomGene, "min_negative_score", fake_min_negative_score)
    monkeypatch.setattr(RandomGene, "node_re_gen_max", fake_node_re_gen_max)
    monkeypatch.setattr(RandomGene, "max_segments", fake_max_segments)
    monkeypatch.setattr(RandomGene, "number_of_children_range", fake_number_of_children_range)
    monkeypatch.setattr(RandomGene, "random_normal_within_range", fake_random_normal_within_range)
    monkeypatch.setattr(RandomGene, "random_segment_colour", fake_random_segment_colour)

    # Create dummy design and parent objects.
    class DummyParent:
        def __init__(self):
            self.absolute_angle = 0
            self.children = []
            self.end_thickness = [1]
            self.start = (0, 0)
            self.segment_type = RandomGene.SegmentType.LINE
    class DummyDesign:
        def __init__(self):
            self.children = []
        def render_design(self, show=False):
            pass
        def get_all_nodes(self):
            return []
    dummy_design = DummyDesign()
    dummy_parent = DummyParent()

    # Call random_gene_node.
    success, score = random_gene_node(dummy_design, dummy_parent, "blue", segment_number=1, depth=0)

    # Our fake_check_new_segments_negative_score returns 20, so expect score = 20.
    assert success is True, "Expected random_gene_node to succeed."
    assert score == 20, f"Expected score 20, got {score}"
    # Parent's children should have one new node.
    assert len(dummy_parent.children) == 1, f"Expected parent's children to contain 1 node, got {len(dummy_parent.children)}"

    # Force rendering on the new node.
    new_node = dummy_parent.children[0]
    if not hasattr(new_node, "points_array"):
        new_node.render(np.array([new_node.start]), new_node.absolute_angle, new_node.colour, new_node.end_thickness)
    assert hasattr(new_node, "points_array"), "New node should have a 'points_array' attribute after rendering."
    assert new_node.points_array.size > 0, "New node's points_array should not be empty."

# --- Test for random_gene ---
@pytest.mark.parametrize("gene_n", [2, 4, 7])

def test_random_gene(gene_n, monkeypatch):
    import RandomGene as RG
    import random

    # Patch helper functions in RG for random_gene.
    monkeypatch.setattr(RG, "random_segment_colour", fake_random_segment_colour)
    monkeypatch.setattr(RG, "make_eyeliner_wing", fake_make_eyeliner_wing)
    monkeypatch.setattr(RG, "EyelinerDesign", fake_EyelinerDesign)
    monkeypatch.setattr(RG, "is_outside_face_area", fake_is_outside_face_area)
    monkeypatch.setattr(RG, "is_in_eye", fake_is_in_eye)
    monkeypatch.setattr(RG, "random_normal_within_range", fake_random_normal_within_range_for_gene)
    monkeypatch.setattr(RG, "analyse_negative", fake_analyse_negative)
    monkeypatch.setattr(RG, "set_prev_end_thickness_array", fake_set_prev_end_thickness_array)
    monkeypatch.setattr(RG, "set_prev_array", fake_set_prev_array)
    monkeypatch.setattr(RG, "n_of_children_decreasing_likelihood", lambda *args, **kwargs: 0)
    monkeypatch.setattr(RG, "initial_gene_pool_size", fake_initial_gene_pool_size)
    monkeypatch.setattr(RG, "min_negative_score", fake_min_negative_score)

    # Force random.random() to return 0.5.
    monkeypatch.setattr(random, "random", lambda: 0.5)

    # Call random_gene.
    design = RG.random_gene(gene_n)

    # Check that design is an EyelinerDesign with a root.
    assert hasattr(design, "root"), "Design should have a 'root' attribute."
    root = design.root
    # Render the root if needed.
    if not hasattr(root, "points_array"):
        root.render(np.array([root.start]), root.absolute_angle, root.colour, root.end_thickness)
    assert hasattr(root, "points_array"), "Root segment should have a 'points_array' attribute after rendering."
    assert root.points_array.size > 0, "Root segment's points_array should not be empty."
