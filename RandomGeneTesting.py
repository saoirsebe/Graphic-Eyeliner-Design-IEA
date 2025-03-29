import pytest
import RandomGene
from RandomGene import random_gene_node, random_gene
import numpy as np

# Our fake random function for testing: returns the mean.
def fake_random_normal_within_range(mean, std_dev, value_range):
    return mean


@pytest.mark.parametrize(
    "segment_number, branch_length, max_segments, base_mean, std_dev, value_range, expected",
    [
        # Test 1: Early segment, shallow branch.
        # global_decay_factor = (1 - 1/100)**1.6 ≈ 0.99**1.6 ≈ 0.984,
        # branch_decay_factor = 1 - (1/5) = 0.8,
        # decay_factor ≈ 0.984 * 0.8 = 0.7872,
        # mean ≈ 4 * 0.7872 ≈ 3.1488, round(3.1488) = 3.
        (1, 1, 100, 4, 1, (0, 10), 3),

        # Test 2: Late segment: global_decay_factor becomes 0.
        (100, 1, 100, 4, 1, (0, 10), 0),

        # Test 3: Deep branch: branch_decay_factor becomes 0 (branch_length == average_branch_length, here 5).
        (1, 5, 100, 4, 1, (0, 10), 0),

        # Test 4: Mid-range values.
        # For segment_number=50, branch_length=2, max_segments=100, base_mean=6:
        # global_decay_factor = (1 - 50/100)**1.6 = (0.5)**1.6 ≈ 0.330,
        # branch_decay_factor = 1 - (2/5) = 0.6,
        # decay_factor ≈ 0.330 * 0.6 = 0.198,
        # mean ≈ 6 * 0.198 ≈ 1.188, round(1.188) = 1.
        (50, 2, 100, 6, 1, (0, 10), 1),

        # Test 5: Early segment but deeper branch.
        # For segment_number=1, branch_length=3, max_segments=100, base_mean=8:
        # global_decay_factor ≈ (1 - 1/100)**1.6 ≈ 0.984,
        # branch_decay_factor = 1 - (3/5) = 0.4,
        # decay_factor ≈ 0.984 * 0.4 ≈ 0.3936,
        # mean ≈ 8 * 0.3936 ≈ 3.1488, round(3.1488) = 3.
        (1, 3, 100, 8, 1, (0, 10), 3),
    ]
)
def test_n_of_children_decreasing_likelihood(segment_number, branch_length, max_segments, base_mean, std_dev,
                                             value_range, expected, monkeypatch):
    # Patch the random_normal_within_range in the RandomGeneTesting module.
    monkeypatch.setattr(RandomGene, "random_normal_within_range", fake_random_normal_within_range)

    result = RandomGene.n_of_children_decreasing_likelihood(segment_number, branch_length, max_segments,
                                                                   base_mean, std_dev, value_range)
    assert result == expected, (
        f"Expected {expected} children, got {result} for segment_number={segment_number}, "
        f"branch_length={branch_length}, base_mean={base_mean}"
    )

# --- Dummy Classes and Fake Helper Functions for Testing random_gene_node ---

class DummySegment:
    def __init__(self):
        self.colour = "blue"
        self.children = []
        self.start = (0, 0)
        self.absolute_angle = 0
        self.end_thickness = [1]
        self.segment_type = "LINE"
        self.points = "LINE"
    def render(self, *args, **kwargs):
        pass

def fake_random_segment(prev_colour=None, segment_start=None):
    return DummySegment()

def fake_set_prev_end_thickness_array(parent):
    # For testing, simply return the parent's end_thickness.
    return parent.end_thickness

def fake_set_prev_array(parent):
    # Return an array containing parent's start.
    return np.array([parent.start])

def fake_check_new_segments_negative_score(design, node):
    # Return a fixed score above our fake minimum fitness so that regeneration does not occur.
    return 0

def fake_n_of_children_decreasing_likelihood(segment_number, depth, max_segments, base_mean, std_dev, value_range):
    # For testing, force no children (to stop recursion).
    return 0

# Define fake global values for testing.
fake_min_fitness_score = -1
fake_node_re_gen_max = 12
fake_max_segments = 20
fake_number_of_children_range = (0, 3)

# --- Test for random_gene_node ---
def test_random_gene_node(monkeypatch):
    # Patch helper functions and globals in RandomGene.
    import RandomGene

    monkeypatch.setattr(RandomGene, "random_segment", fake_random_segment)
    monkeypatch.setattr(RandomGene, "set_prev_end_thickness_array", fake_set_prev_end_thickness_array)
    monkeypatch.setattr(RandomGene, "set_prev_array", fake_set_prev_array)
    monkeypatch.setattr(RandomGene, "check_new_segments_negative_score", fake_check_new_segments_negative_score)
    monkeypatch.setattr(RandomGene, "n_of_children_decreasing_likelihood", fake_n_of_children_decreasing_likelihood)
    monkeypatch.setattr(RandomGene, "min_fitness_score", fake_min_fitness_score)
    monkeypatch.setattr(RandomGene, "node_re_gen_max", fake_node_re_gen_max)
    monkeypatch.setattr(RandomGene, "max_segments", fake_max_segments)
    monkeypatch.setattr(RandomGene, "number_of_children_range", fake_number_of_children_range)

    # Create dummy design and parent objects.
    class DummyParent:
        def __init__(self):
            self.absolute_angle = 0
            self.children = []
            self.end_thickness = [1]
            self.start = (0, 0)
            self.segment_type = "LINE"
    class DummyDesign:
        def __init__(self):
            self.children = []
        def render_design(self, show=False):
            pass

    dummy_design = DummyDesign()
    dummy_parent = DummyParent()

    # Call the function under test.
    success, score = random_gene_node(dummy_design, dummy_parent, "blue", segment_number=1, depth=0)

    # We expect success True and the score to be the value from fake_check_new_segments_negative_score (15).
    assert success is True, "Expected random_gene_node to succeed."
    assert score == 15, f"Expected score 15, got {score}"
    # Also, the parent's children should have one node appended.
    assert len(dummy_parent.children) == 1, f"Expected parent's children to contain 1 node, got {len(dummy_parent.children)}"

# --- Dummy Classes and Fake Helper Functions for Testing random_gene ---

# For random_gene, we need a fake random_segment that returns a DummySegment with a segment_type.
def fake_random_segment_for_gene(*args, **kwargs):
    # Return a DummySegment that has a segment_type (set it to "LINE" by default).
    seg = DummySegment()
    seg.segment_type = "LINE"
    return seg

def fake_random_segment_colour(prev_colour=None):
    return "blue"

def fake_make_eyeliner_wing(random_colour):
    # Return a dummy segment.
    return DummySegment()

# Dummy EyelinerDesign: simply wraps a segment as its root.
class DummyEyelinerDesign:
    def __init__(self, segment):
        self.root = segment
    def render_design(self, show=False):
        pass

def fake_EyelinerDesign(segment):
    return DummyEyelinerDesign(segment)

def fake_is_outside_face_area(root_node):
    # For testing, force the root_score branch that sets root_score = 2 * min_fitness_score.
    return True

def fake_is_in_eye(root_node):
    return 0

def fake_analyse_negative(design):
    # Return a value equal to min_fitness_score so that analysis passes.
    return fake_min_fitness_score

def fake_random_normal_within_range_for_gene(mean, stddev, value_range):
    return mean

# Set a fake initial gene pool size.
fake_initial_gene_pool_size = 9

# --- Test for random_gene ---
# We parameterize by gene_n to test different branches (if any).
@pytest.mark.parametrize("gene_n", [2, 4, 7])
def test_random_gene(gene_n, monkeypatch):
    import RandomGene as RG
    import random

    # Patch helper functions in RandomGene.
    monkeypatch.setattr(RG, "random_segment", fake_random_segment_for_gene)
    monkeypatch.setattr(RG, "random_segment_colour", fake_random_segment_colour)
    monkeypatch.setattr(RG, "make_eyeliner_wing", fake_make_eyeliner_wing)
    monkeypatch.setattr(RG, "EyelinerDesign", fake_EyelinerDesign)
    monkeypatch.setattr(RG, "is_outside_face_area", fake_is_outside_face_area)
    monkeypatch.setattr(RG, "is_in_eye", fake_is_in_eye)
    monkeypatch.setattr(RG, "random_normal_within_range", fake_random_normal_within_range_for_gene)
    monkeypatch.setattr(RG, "analyse_negative", fake_analyse_negative)
    # Patch the helper functions for random_gene_node that are used by random_gene.
    monkeypatch.setattr(RG, "set_prev_end_thickness_array", fake_set_prev_end_thickness_array)
    monkeypatch.setattr(RG, "set_prev_array", fake_set_prev_array)
    # Also patch random.random() so that its value is fixed.
    monkeypatch.setattr(random, "random", lambda: 0.5)
    # Patch n_of_children_decreasing_likelihood to return 0 so that no recursion occurs.
    monkeypatch.setattr(RG, "n_of_children_decreasing_likelihood", lambda *args, **kwargs: 0)
    # Patch global variables.
    monkeypatch.setattr(RG, "initial_gene_pool_size", fake_initial_gene_pool_size)
    monkeypatch.setattr(RG, "min_fitness_score", fake_min_fitness_score)

    # Call random_gene.
    design = random_gene(gene_n)

    # Check that the returned design is an EyelinerDesign (our dummy) and has a root.
    assert hasattr(design, "root"), "Design should have a 'root' attribute."
    # Check that the root's colour is as expected.
    assert design.root.colour == "blue", f"Expected root colour 'blue', got '{design.root.colour}'"
    # Check that the design's render_design method exists.
    assert hasattr(design, "render_design"), "Design should have a 'render_design' method."