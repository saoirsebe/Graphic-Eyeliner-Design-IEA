
from A import *
from AnalyseDesign import check_design_overlaps, check_overlaps, analyse_negative, \
    analyse_positive, \
    check_segment_overlaps, is_in_eye, is_outside_face_area, check_new_segments_negative_score
from BreedingMechanism import compare_designs
from EyelinerDesign import EyelinerDesign
from Segments import create_segment, random_segment, set_prev_end_thickness_array, make_eyeliner_wing, \
    random_segment_colour


def n_of_children_decreasing_likelihood(segment_number, branch_length, max_segments, base_mean, std_dev, value_range):
    """
    Determines the number of children for a node based on:
    - Segment number (global position in tree).
    - Branch length (local depth within a branch).
    - A decaying mean to reduce the number of children as branch length increases.

    Parameters:
    - segment_number: The current segment number (1-based index, global).
    - branch_length: The current depth in the branch (1-based index, local).
    - max_segments: The global segment number after which children are unlikely.
    - base_mean: The initial mean for the random normal distribution.
    - std_dev: The standard deviation for the random normal distribution.
    - value_range: A tuple (min_children, max_children).

    Returns:
    - n_of_children: The calculated number of children.
    """
    # Global decay: Reduce likelihood of children based on the segment number
    global_decay_power = 1.6  # Adjust this to control how quickly the number of children drops
    global_decay_factor = max(0, (1 - (segment_number / max_segments)) ** global_decay_power)  # Reduces from 1 to 0

    # Branch decay: Reduce likelihood of children as branch depth increases
    branch_decay_factor = max(0, 1 - (branch_length / average_branch_length))  # Target average branch length ~3.5

    # Combined decay: Both global and branch decay affect the mean
    decay_factor = global_decay_factor * branch_decay_factor
    mean = base_mean * decay_factor  # Adjust mean with decay

    # Generate the number of children
    return round(random_normal_within_range(mean, std_dev, value_range))


def random_gene_node(design, parent, prev_colour, segment_number=1, depth=0):
    #Create new node, render, check overlaps with the other segments in design, if new_node overlaps more than min_fitness_score then re-generate random new_node
    new_node = random_segment(prev_colour=prev_colour)
    prev_end_thickness_array = set_prev_end_thickness_array(parent)
    prev_array = set_prev_array(parent)
    prev_angle = parent.absolute_angle
    new_node.render(prev_array, prev_angle, prev_colour, prev_end_thickness_array)
    """
    #Print FOR TESTING:
    fig, ax_n = plt.subplots(figsize=(3, 3))
    parent.children.append(new_node)
    fig = design.render_design()
    fig.show()
    plt.close(fig)
    parent.children.remove(new_node)
    """

    regen_count = 0

    new_segment_score = check_new_segments_negative_score(design, new_node)
    while new_segment_score < min_fitness_score and regen_count < node_re_gen_max:
        new_node = random_segment( prev_colour=prev_colour)
        new_node.render(prev_array, prev_angle, prev_colour, prev_end_thickness_array)
        """
        # Print FOR TESTING:
        fig, ax_n = plt.subplots(figsize=(3, 3))
        parent.children.append(new_node)
        fig = design.render_design()
        fig.show()
        plt.close(fig)
        parent.children.remove(new_node)
        """
        regen_count+=1
        new_segment_score = check_new_segments_negative_score(design, new_node)
    if regen_count >=node_re_gen_max:
        return False, min_fitness_score *2

    parent.children.append(new_node)
    n_of_children = n_of_children_decreasing_likelihood(segment_number, depth, max_segments, 1.6,0.6, number_of_children_range)
    prev_colour = new_node.colour
    depth+=1

    for i in range(n_of_children):
        segment_number += 1
        success,child_score = random_gene_node(design, new_node, prev_colour, segment_number=segment_number, depth=depth)
        new_segment_score += child_score
        if not success or new_segment_score < min_fitness_score:
            return False , min_fitness_score *2  # Propagate failure if child generation fails.

    return True, new_segment_score

def random_gene(gene_n):
    success = False
    while not success:
        success = True
        root_score =-1
        while root_score < 0:
            #The first 2 thirds of the initial population start at the corner of the eye, the second third starts as an eyeliner wing, last 1/3 is random
            if initial_gene_pool_size/3 < gene_n <= 2* (initial_gene_pool_size / 3):
                random_colour = "black" if random.random() < 0.6 else random_segment_colour()
                design = EyelinerDesign(make_eyeliner_wing(random_colour))
            elif gene_n <= 2 *(initial_gene_pool_size/3):
                design = EyelinerDesign(random_segment(segment_start=eye_corner_start))
            else:
                design = EyelinerDesign(random_segment())

            root_node = design.root
            prev_colour = root_node.colour
            segment_number = 0
            prev_end_thickness_array = np.array([root_node.end_thickness])
            root_node.render(np.array([root_node.start]), 0, prev_colour, prev_end_thickness_array)
            """
            fig, ax_n = plt.subplots(figsize=(3, 3))
            fig = design.render_design()
            fig.show()
            plt.close(fig)
            """

            if is_outside_face_area(root_node):
                root_score = 2*min_fitness_score
            else:
                root_score = -is_in_eye(root_node)

        n_of_children = round(random_normal_within_range(1.2, 0.2, number_of_children_range))
        total_score = root_score
        for i in range(n_of_children):
            segment_number +=1
            success, child_score = random_gene_node(design, root_node, prev_colour, segment_number=segment_number, depth=0)
            total_score+=child_score
            #print("child_score", child_score)
            #print("Total score:",total_score)
            if not success or total_score < min_fitness_score:
                success = False

        design.render_design(show = False)
        if analyse_negative(design)<min_fitness_score:
            success = False

        if success:
            #print(f"success {gene_n}")
            #print("Score:",total_score)
            return design

""""""
"""
#design = random_gene(1)
design =random_gene(170)
#design = random_gene(190)
#fig = design.render_design()
#fig.show()


positive_score = analyse_positive(design, True)
segments = design.get_all_nodes()
print("Positive Score:", positive_score)

negative_score = analyse_negative(design, True)
print("analyse_negative score:", negative_score)
"""
"""
print("New design:")
design2 = design.mutate_design(0.05)
fig = design2.render_design()

difference = compare_designs(design, design2)
print("Difference:", difference)
fig.show()
"""





