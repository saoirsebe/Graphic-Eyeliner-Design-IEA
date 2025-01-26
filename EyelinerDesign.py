import copy
import random
from conda.common.configuration import raise_errors
from AnalyseDesign import analyse_negative
from Segments import *
from A import max_segments

class EyelinerDesign:   #Creates overall design, calculates start points, renders each segment by calling their render function
    def __init__(self, root_node):
        self.root = root_node
        self.n_of_lines= 0
        self.n_of_stars= 0
        self.tree_size= 0

    def get_all_nodes(self, node=None, nodes_list=None):
        """
        Collects all nodes in the tree in depth-first order.

        Parameters:
        - node: The current node (default is the root).
        - nodes_list: The list of nodes being collected.

        Returns:
        - list: A list of all TreeNode objects in depth-first order.
        """
        if nodes_list is None:
            nodes_list = []

        if node is None:
            node = self.root

        # Add the current node
        nodes_list.append(node)

        # Recursively collect nodes from children
        for child in node.children:
            self.get_all_nodes(child, nodes_list)

        self.tree_size = len(nodes_list)

        return nodes_list

    def add_segment_at(self,segment,index,is_branch):
        nodes_list = self.get_all_nodes()

        if index < 0 or index > self.tree_size -1:
            raise IndexError("Index out of range. Cannot add segment at this position.")

        parent_node = nodes_list[index]

        if is_branch:
            parent_node.add_child_segment(segment)
        else:
            parent_children = parent_node.get_children
            parent_node.children = [segment]
            segment.children.append(parent_children)

    def is_descendant(self, node, potential_descendant):
        """
        Checks if a node is a descendant of another node.

        Parameters:
        - node: The potential ancestor node.
        - potential_descendant: The node to check.

        Returns:
        - bool: True if the node is a descendant, False otherwise.
        """
        if potential_descendant in node.children:
            return True
        for child in node.children:
            if self.is_descendant(child, potential_descendant):
                return True
        return False

    def find_parent(self, node, target_node):
        """
        Finds the parent of a given node in the tree.
        """
        for child in node.children:
            if child == target_node:
                return node
            result = self.find_parent(child, target_node)
            if result:
                return result
        return None

    def move_subtree(self, subtree_root, original_parent):
        """
        Randomly moves a subtree to a new location in the tree.
        """
        # Get all nodes in the tree
        nodes_list = self.get_all_nodes()

        # Ensure there are at least two nodes (root and another node)
        if len(nodes_list) < 2:
            print("Not enough nodes to perform a move.")
            return

        original_parent.children.remove(subtree_root)

        # Randomly select a new parent node (cannot be the subtree root itself or any of its descendants)
        valid_nodes = [node for node in nodes_list if
                       node != subtree_root and not self.is_descendant(node, subtree_root)]
        if not valid_nodes:
            print("No valid node to move the subtree to.")
            return

        new_parent = random.choice(valid_nodes)

        # Attach the subtree to the new parent
        new_parent.add_child(subtree_root)

    def render_node(self, ax_n, node, prev_array, prev_angle, prev_colour, prev_end_thickness_array):
        node.render(ax_n, prev_array, prev_angle,prev_colour,prev_end_thickness_array)
        prev_segment = node
        if prev_segment.segment_type == SegmentType.STAR:
            # if segment is a star then pass in the arm points that the next segment should start at:
            prev_array = prev_segment.arm_points_array
        else:
            prev_array = prev_segment.points_array
        prev_angle = prev_segment.absolute_angle
        prev_colour = prev_segment.colour
        if prev_segment.segment_type == SegmentType.LINE:
            prev_end_thickness_array = prev_segment.thickness_array
        else:
            prev_end_thickness_array = np.array(prev_segment.end_thickness)

        for child in node.children:
            self.render_node(ax_n, child, prev_array, prev_angle, prev_colour, prev_end_thickness_array)

    def render_design(self):
        """Render the eyeliner design"""
        root_node = self.root  # Start from the root

        fig, ax_n = plt.subplots(figsize=(3, 3))
        draw_eye_shape(ax_n)
        prev_array = np.array([root_node.start])
        prev_angle = 0
        prev_colour = root_node.colour
        prev_end_thickness_array = root_node.end_thickness

        self.render_node(ax_n, root_node, prev_array, prev_angle, prev_colour, prev_end_thickness_array)

        return fig

    def mutate_node(self,node,mutation_rate=0.05):
        if random.random() < mutation_rate:
            node.mutate(mutation_rate)

        #Random chance of loosing a child
        if len(node.children) > 1:
            delete_child_chance = mutation_rate
        elif len(node.children) == 1:
            delete_child_chance = mutation_rate //2
        else:
            delete_child_chance = 0

        if random.random() < delete_child_chance:
            delete_child_index = random.randint(0,len(node.children)-1)
            node.children.pop(delete_child_index)

        for child in node.children:
            # Random chance of swapping position:
            if random.random() < mutation_rate:
                self.move_subtree(child,node)

            self.mutate_node(child, mutation_rate)

    def mutate_self(self, mutation_rate=0.05):
        node = self.root
        self.mutate_node(node,mutation_rate)
        for child in node.children:
            self.mutate_node(child,mutation_rate)

        # Random chance of adding in a new segment:
        if np.random.random() < mutation_rate:
            new_segment = random_segment(False)
            nodes_list = self.get_all_nodes()
            is_branch = False if random.random() < 0.7 else True #If the new segment branches off a segment or is placed in-between segments
            self.add_segment_at(new_segment, np.random.randint(0, len(nodes_list)-1),is_branch)

        return self

    def mutate(self,mutation_rate=0.05):
        new_gene = copy.deepcopy(self)
        new_gene= new_gene.mutate_self(mutation_rate)
        fig=new_gene.render_design()
        plt.close(fig)
        overlap_score = analyse_negative(new_gene)
        print("overlap_score", overlap_score)
        while overlap_score<=min_fitness_score:
            new_gene = copy.deepcopy(self)
            new_gene = new_gene.mutate_self(mutation_rate)
            fig = new_gene.render_design()
            plt.close(fig)
            overlap_score = analyse_negative(new_gene)
            print("overlap_score", overlap_score)
        return new_gene

def random_normal_within_range(mean, stddev, value_range):
    while True:
        # Generate a number using normal distribution
        value = random.gauss(mean, stddev)
        # Keep it within the specified range
        if value_range[0] <= value <= value_range[1]:
            return value

def random_from_two_distributions(mean1, stddev1, mean2, stddev2, value_range, prob1=0.5):
    while True:
        # Choose which distribution to use
        if random.random() < prob1:
            value = random.gauss(mean1, stddev1)
        else:  # Otherwise, use the second distribution
            value = random.gauss(mean2, stddev2)

        if value_range[0] <= value <= value_range[1]:
            return value

def random_segment(eyeliner_wing = False, prev_colour=None,segment_number = 0, segment_start=(random.uniform(*start_x_range), random.uniform(*start_y_range))):
    r = random.random()
    #If we want to start with an eyeliner wing, the first two segments must be lines:
    if eyeliner_wing:
        new_segment_type = SegmentType.LINE
    else:
        if r < 0.65:
            new_segment_type = SegmentType.LINE
        else:
            new_segment_type = SegmentType.STAR

    #Pick colour, more likely to be the previous colour:
    if prev_colour is None:
        random_colour = random.choice(colour_options)
    else:
        #weights = [0.6 if colour == prev_colour else 0.1 for colour in colour_options]
        #random_colour = random.choices(colour_options, weights=weights, k=1)[0]
        random_colour = prev_colour if random.random() < 0.5 else random.choice(colour_options)

    if new_segment_type == SegmentType.LINE:
        random_start_mode = random.choice(list(StartMode))
        if segment_start == (3, 1.5) and eyeliner_wing: #First segment (line) in eyeliner wing:
            random_relative_angle = random_normal_within_range(22.5, 40, direction_range)
        elif eyeliner_wing and segment_number == 1: #Second segment in eyeliner wing:
            random_relative_angle = random_normal_within_range(157.5, 40, direction_range)
            random_start_mode = StartMode.CONNECT
        elif random_start_mode == StartMode.CONNECT:
            random_relative_angle = random_from_two_distributions(135, 60, 225, 60, direction_range)
        elif random_start_mode == StartMode.CONNECT_MID:
            random_relative_angle = random_from_two_distributions(90,50,270,50, direction_range)
        elif random_start_mode == StartMode.SPLIT:
            random_relative_angle = random_from_two_distributions(90,50,270,50, direction_range)
        else:
            random_relative_angle = random.uniform(*direction_range)

        new_segment = create_segment(
            segment_type=SegmentType.LINE,
            start=segment_start,
            start_mode=random_start_mode,
            length=random.uniform(*length_range),
            relative_angle=random_relative_angle,
            start_thickness=random.uniform(*thickness_range),
            end_thickness=random.uniform(*thickness_range),
            colour=random_colour,
            curviness=0 if random.random() < 0.5 else random_normal_within_range(0.1,0.5,curviness_range),
            curve_direction=random_from_two_distributions(90,40,270,40, direction_range),
            curve_location=random_normal_within_range(0.5,0.25,relative_location_range),
            start_location=random_normal_within_range(0.5,0.25,relative_location_range),
            split_point=random_normal_within_range(0.5,0.25,relative_location_range)
        )
    elif new_segment_type == SegmentType.STAR:
        new_segment = create_segment(
            segment_type=SegmentType.STAR,
            start=segment_start,
            start_mode=random.choice([StartMode.CONNECT, StartMode.JUMP]),
            radius=random_normal_within_range(0.5,0.5,radius_range),
            arm_length=random_normal_within_range(0.75,0.5,arm_length_range),
            num_points=round(random_normal_within_range(5,2, num_points_range)),
            asymmetry=0 if random.random() < 0.6 else random_normal_within_range(2,2,asymmetry_range),
            star_type=random.choice([StarType.STRAIGHT, StarType.CURVED, StarType.FLOWER]),
            end_thickness=random_normal_within_range(2,2,thickness_range),
            relative_angle=random.uniform(*direction_range),
            colour=random_colour,
        )
    return new_segment


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
    global_decay_factor = max(0, 1 - (segment_number / max_segments))  # Reduces from 1 to 0

    # Branch decay: Reduce likelihood of children as branch depth increases
    branch_decay_factor = max(0, 1 - (branch_length / average_branch_length))  # Target average branch length ~3

    # Combined decay: Both global and branch decay affect the mean
    decay_factor = global_decay_factor * branch_decay_factor
    mean = base_mean * decay_factor  # Adjust mean with decay

    # Generate the number of children
    return round(random_normal_within_range(mean, std_dev, value_range))

def random_gene_node(parent,prev_colour,eyeliner_wing=False,segment_number=1,depth=0):
    new_node = random_segment(eyeliner_wing=eyeliner_wing,segment_number=segment_number,prev_colour=prev_colour)
    parent.children.append(new_node)
    n_of_children = n_of_children_decreasing_likelihood(segment_number, depth, max_segments, 1,0.6, number_of_children_range)
    prev_colour = new_node.colour
    depth+=1
    print("n_of_children:", n_of_children)
    for i in range(n_of_children):
        segment_number+=1
        random_gene_node(new_node,prev_colour,segment_number=segment_number,depth=depth)

def random_gene(gene_n,):
    ##The first 2 thirds of the initial population start at the corner of the eye, the second third starts as an eyeliner wing, last 1/3 is random
    if initial_gene_pool_size/3 < gene_n <= 2* (initial_gene_pool_size / 3):
        design = EyelinerDesign(random_segment(eyeliner_wing=True, segment_number=0,segment_start=(3, 1.5)))
        n_of_children = 2
    elif gene_n <= 2 *(initial_gene_pool_size/3):
        design = EyelinerDesign(random_segment(segment_start=(3, 1.5)))
        n_of_children = round(random_normal_within_range(1, 0.5, number_of_children_range))
    else:
        design = EyelinerDesign(random_segment())
        n_of_children = round(random_normal_within_range(1,0.5,number_of_children_range))

    root_node = design.root
    prev_colour = root_node.colour
    segment_number = 0
    for i in range(n_of_children):
        segment_number +=1
        random_gene_node(root_node,prev_colour,eyeliner_wing=True,segment_number=segment_number,depth=0)

    return design

"""
random_design = random_gene(0)
fig = random_design.render()
fig.show()
score = analyse_design_shapes(random_design)
print("Score:", score)
"""