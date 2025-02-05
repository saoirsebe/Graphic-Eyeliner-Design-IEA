import copy
import random
from conda.common.configuration import raise_errors
from AnalyseDesign import analyse_negative, check_overlaps, shape_overlaps, check_segment_overlaps, check_design_overlaps
from Segments import *
import cProfile
from IrregularPolygonSegment import are_points_collinear

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

    def move_subtree(self, subtree_root):
        """
        Randomly moves a subtree to a new location in the tree.
        """
        # Get all nodes in the tree
        nodes_list = self.get_all_nodes()
        # Ensure there are at least two nodes (root and another node)
        if len(nodes_list) < 2:
            print("Not enough nodes to perform a move.")
            return

        original_parent = find_parent(self.root, subtree_root)
        original_parent.children.remove(subtree_root)

        # Randomly select a new parent node (cannot be the subtree root itself or any of its descendants)
        valid_nodes = [node for node in nodes_list if
                       node != subtree_root and not self.is_descendant(node, subtree_root)]
        if not valid_nodes:
            print("No valid node to move the subtree to.")
            return

        new_parent = random.choice(valid_nodes)

        # Attach the subtree to the new parent
        new_parent.add_child_segment(subtree_root)

    def render_node(self, node, prev_array, prev_angle, prev_colour, prev_end_thickness_array, ax_n=None):
        node.render(prev_array, prev_angle,prev_colour,prev_end_thickness_array,ax_n)
        prev_segment = node

        if prev_segment.segment_type == SegmentType.LINE:
            prev_array = prev_segment.points_array
        elif prev_segment.segment_type == SegmentType.STAR:
            # if segment is a star then pass in the arm points that the next segment should start at:
            prev_array = prev_segment.arm_points_array
        elif prev_segment.segment_type == SegmentType.RANDOM_SHAPE:
            prev_array = prev_segment.corners

        prev_angle = prev_segment.absolute_angle
        prev_colour = prev_segment.colour
        if prev_segment.segment_type == SegmentType.LINE:
            prev_end_thickness_array = prev_segment.thickness_array
        else:
            prev_end_thickness_array = np.array(prev_segment.end_thickness)

        for child in node.children:
            self.render_node( child, prev_array, prev_angle, prev_colour, prev_end_thickness_array,ax_n)

    def render_design(self, show = True):
        """Render the eyeliner design"""
        root_node = self.root  # Start from the root
        prev_array = np.array([root_node.start])
        prev_angle = 0
        prev_colour = root_node.colour
        prev_end_thickness_array = root_node.end_thickness

        if show:
            fig, ax_n = plt.subplots(figsize=(3, 3))
            draw_eye_shape(ax_n)
            self.render_node(root_node, prev_array, prev_angle, prev_colour, prev_end_thickness_array,ax_n)
            return fig
        else:
            self.render_node(root_node, prev_array, prev_angle, prev_colour, prev_end_thickness_array)
            return

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

        #Random chance of swapping a subtree position:
        if np.random.random() < mutation_rate and len(self.get_all_nodes())>2:
            nodes_list = self.get_all_nodes()
            self.move_subtree(random.choice(nodes_list[1:]))
        return self

    def mutate(self,mutation_rate=0.05):
        new_gene = copy.deepcopy(self)
        new_gene= new_gene.mutate_self(mutation_rate)
        new_gene.render_design(show=False)
        overlap_score = analyse_negative(new_gene)
        print("overlap_score", overlap_score)
        while overlap_score<=min_fitness_score:
            new_gene = copy.deepcopy(self)
            new_gene = new_gene.mutate_self(mutation_rate)
            new_gene.render_design(show=False)
            overlap_score = analyse_negative(new_gene)
            print("overlap_score", overlap_score)
        return new_gene


"""
random_design = random_gene(0)
fig = random_design.render()
fig.show()
score = analyse_design_shapes(random_design)
print("Score:", score)
"""
def random_curviness(mean=0.3,stddev=0.25):
    return random_normal_within_range(mean, stddev, curviness_range)
def random_curve_direction():
    return random_from_two_distributions(90, 40, 270, 40, direction_range)
def random_curve_location():
    return random_normal_within_range(0.5, 0.25, relative_location_range)
def angle_from_center(center, point):
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    return math.atan2(dy, dx)  # atan2 returns the angle in radians

def random_lines_corners_list(n_of_corners):
    collinear=True
    #Check if points are collinear within tolerance
    while collinear:
        corners = np.array(
            [(random.uniform(*edge_initialisation_range), random.uniform(*edge_initialisation_range))
             for i in range(n_of_corners)])
        collinear = are_points_collinear(corners)

    lines_list = [(IrregularPolygonEdgeSegment(random_curviness(0.45, 0.15), random_normal_within_range(0.5, 0.15, relative_location_range)))
                  for i in range(n_of_corners)]
    centroid = (sum(point[0] for point in corners) / n_of_corners, sum(point[1] for point in corners) / n_of_corners)
    sorted_corners = sorted(corners, key=lambda point: angle_from_center(centroid, point))

    return sorted_corners, lines_list

def random_random_shape():
    new_shape_overlaps = 10
    while new_shape_overlaps>max_shape_overlaps:
        random_colour = random.choice(colour_options)
        segment_start=(random.uniform(*start_x_range), random.uniform(*start_y_range))
        n_of_corners = round(random_normal_within_range(5,1.5, num_points_range))
        corners, lines = random_lines_corners_list(n_of_corners)
        new_segment = create_segment(
            segment_type=SegmentType.RANDOM_SHAPE,
            start=segment_start,
            start_mode=random.choice([StartMode.CONNECT, StartMode.JUMP]),
            end_thickness=random_normal_within_range(2, 2, thickness_range),
            relative_angle=random.uniform(*direction_range),
            colour=random_colour,
            bounding_size=(random.uniform(*random_shape_size_range), random.uniform(*random_shape_size_range)),
            corners=corners,
            lines_list= lines
        )
        prev_array = np.array([new_segment.start])
        prev_angle = 0
        prev_colour = new_segment.colour
        prev_end_thickness = new_segment.end_thickness
        new_segment.render(prev_array, prev_angle, prev_colour, prev_end_thickness)
        new_shape_overlaps = shape_overlaps(new_segment.lines_list)

    return new_segment

"""
fig, ax_n = plt.subplots(figsize=(3, 3))
shape = random_random_shape()
prev_array = np.array([shape.start])
prev_angle = 0
prev_colour = shape.colour
prev_end_thickness= shape.end_thickness
shape.render(prev_array, prev_angle, prev_colour, prev_end_thickness, ax_n)

fig.show()



#design = EyelinerDesign(random_random_shape())
#design.render_design()
line = IrregularPolygonEdgeSegment(0.7, 0.2)
start= np.array((-1.0,2.0))
end = np.array((3.0,2.0))
line.render((0,0), start, end, 'green', 3, ax_n)

ax_n.set_xlim(-2, 4)  # Set x-axis limits
ax_n.set_ylim(0, 3)  # Set y-axis limits

line = LineSegment(SegmentType.LINE, (-1,1), StartMode.JUMP, 4, 0, 3, 3, 'red', 0.5, False, 0.5, 0, 0)
line.render(np.array([line.start]), 0, 'red', line.end_thickness, ax_n)


cProfile.run('random_random_shape()')
"""

fig, ax_n = plt.subplots(figsize=(3, 3))
line = LineSegment(SegmentType.LINE, (5,1), StartMode.JUMP, 4, 20, 4, 4, 'red', 0.5, False, 0.5, 0, 0)
line2 = LineSegment(SegmentType.LINE, (4,1), StartMode.CONNECT_MID, 4, 50, 2, 2, 'blue', 0.5, False, 0.5, 0.5, 0)
line3 = LineSegment(SegmentType.LINE, (5,1), StartMode.SPLIT, 4, 90, 4, 4, 'red', 0.5, False, 0.5, 0, 0.5)



design = EyelinerDesign(line)
line.add_child_segment(line2)
line2.add_child_segment(line3)
fig = design.render_design(ax_n)
fig.show()

negative_score = analyse_negative(design)
print("negative_score:",negative_score)
