import copy
import random
from conda.common.configuration import raise_errors
from AnalyseDesign import analyse_negative, analyse_positive, check_overlaps, fix_overlaps_shape_overlaps, check_segment_overlaps, check_design_overlaps, is_outside_face_area
from Segments import *
import cProfile
from IrregularPolygonSegment import are_points_collinear
from PIL import Image, ImageTk

class EyelinerDesign:   #Creates overall design, calculates start points, renders each segment by calling their render function
    def __init__(self, root_node):
        self.root = root_node
        self.n_of_lines= 0
        self.n_of_stars= 0
        self.tree_size= 0
        self.image_path = "female-face-drawing-template-one-eye.jpg"
        self.eye_image = Image.open(self.image_path)

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
            parent_children = parent_node.children #parent_node.get_children
            parent_node.children = [segment]
            segment.children.extend(parent_children)

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

    def find_the_parent(self, node, target_node):
        """
        Finds the parent of a given node in the tree.
        """
        for child in node.children:
            if child == target_node:
                return node
            result = self.find_the_parent(child, target_node)
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

        original_parent = self.find_the_parent(self.root, subtree_root)
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

    def render_node(self, node, prev_array, prev_angle, prev_colour, prev_end_thickness_array, scale = 1, ax_n=None):

        try:
            node.render(prev_array, prev_angle, prev_colour, prev_end_thickness_array, scale = scale, ax_n=ax_n)
        except AttributeError as e:
            print(f"AttributeError occurred: {e}")
            print(f"node object: {node}")

        prev_segment = node

        if prev_segment.segment_type == SegmentType.LINE:
            prev_array = prev_segment.points_array
        elif prev_segment.segment_type == SegmentType.STAR:
            # if segment is a star then pass in the arm points that the next segment should start at:
            prev_array = prev_segment.arm_points_array
        elif prev_segment.segment_type == SegmentType.IRREGULAR_POLYGON:
            prev_array = prev_segment.to_scale_corners

        prev_angle = prev_segment.absolute_angle
        prev_colour = prev_segment.colour
        if prev_colour == False:
            print("prev_colour == False")
        if prev_segment.segment_type == SegmentType.LINE:
            prev_end_thickness_array = prev_segment.thickness_array
        else:
            prev_end_thickness_array = np.array(prev_segment.end_thickness)

        for child in node.children:
            self.render_node( child, prev_array, prev_angle, prev_colour, prev_end_thickness_array, scale = scale,ax_n = ax_n)

    def render_design(self, scale = 1, show = True):
        """Render the eyeliner design"""
        root_node = self.root  # Start from the root
        prev_array = np.array([root_node.start])
        prev_angle = 0
        prev_colour = root_node.colour
        if prev_colour == False:
            print("root colour == False")
        prev_end_thickness_array = root_node.end_thickness

        if show:
            fig, ax_n = plt.subplots(figsize=(self.eye_image.width / 100, self.eye_image.height / 100), dpi=100)

            flipped_img = np.flipud(self.eye_image)
            ax_n.imshow(flipped_img)
            ax_n.invert_yaxis()
            #ax_n.plot(face_end_x_values, face_end_y_values, label=f"$y = 0.5x^2$", color="b")
            #generate_middle_curve_lines(ax_n)
            #generate_eyeliner_curve_lines(ax_n)
            #fig, ax_n = plt.subplots(figsize=(3, 3))
            #draw_eye_shape(ax_n)
            self.render_node(root_node, prev_array, prev_angle, prev_colour, prev_end_thickness_array, scale = scale, ax_n = ax_n)
            return fig
        else:
            self.render_node(root_node, prev_array, prev_angle, prev_colour, prev_end_thickness_array, scale = scale)
            return

    def mutate_node(self,node,mutation_rate=0.05):
        if random.random() < mutation_rate:
            node.mutate(mutation_rate)

        for child in node.children:
            self.mutate_node(child, mutation_rate)


    def mutate_self(self, mutation_rate=0.05):
        nodes_list = self.get_all_nodes()

        # Random chance of deleting a segment:
        if len(nodes_list) >1 and np.random.random() < mutation_rate:
            #Assign higher weight to later nodes:
            delete_node = random.choices(nodes_list[1:], weights=range(1, len(nodes_list)), k=1)[0]
            original_parent = self.find_the_parent(self.root, delete_node)
            original_parent.children.remove(delete_node)
            if len(delete_node.children) > 0:
                original_parent.children.extend(delete_node.children)

            nodes_list = self.get_all_nodes()

        #Random chance of swapping a subtree position:
        if np.random.random() < mutation_rate and len(nodes_list)>2:
            self.move_subtree(random.choice(nodes_list[1:]))
            nodes_list = self.get_all_nodes()

        node = self.root
        self.mutate_node(node,mutation_rate)
        for child in node.children:
            self.mutate_node(child,mutation_rate)

        len_nodes_list = len(nodes_list)

        #Random chance of adding in a new segment:
        if len_nodes_list <max_segments and np.random.random() < mutation_rate:
            nodes_list = self.get_all_nodes()
            is_branch = False if random.random() < 0.7 else True #If the new segment branches off a segment or is placed in-between segments
            if len_nodes_list>1:
                new_parent_int = np.random.randint(0, len_nodes_list - 1)
                new_parent_colour = nodes_list[new_parent_int].colour
                new_segment = random_segment(prev_colour=new_parent_colour)
                self.add_segment_at(new_segment, new_parent_int,is_branch)
            else:
                new_segment = random_segment()
                self.add_segment_at(new_segment,0,is_branch)

        return self

    def mutate_design(self,mutation_rate=0.05):
        old_gene = copy.deepcopy(self)
        new_gene= old_gene.mutate_self(mutation_rate)
        new_gene.render_design(show=False)
        overlap_score = analyse_negative(new_gene)
        #print("overlap_score", overlap_score)
        while overlap_score<=min_fitness_score:
            old_gene = copy.deepcopy(self)
            new_gene = old_gene.mutate_self(mutation_rate)
            new_gene.render_design(show=False)
            overlap_score = analyse_negative(new_gene)
            #print("overlap_score", overlap_score)
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
            [(random.uniform(*corner_initialisation_range), random.uniform(*corner_initialisation_range))
             for i in range(n_of_corners)])
        collinear = are_points_collinear(corners)

    lines_list = [(IrregularPolygonEdgeSegment(random_curviness(0.45, 0.15), random_normal_within_range(0.5, 0.15, relative_location_range)))
                  for i in range(n_of_corners)]
    centroid = (sum(point[0] for point in corners) / n_of_corners, sum(point[1] for point in corners) / n_of_corners)
    sorted_corners = sorted(corners, key=lambda point: angle_from_center(centroid, point))

    return sorted_corners, lines_list

"""
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
"""
fig, ax_n = plt.subplots(figsize=(3, 3))
shape = random_irregular_polygon(ax_n)
prev_array = np.array([shape.start])
prev_angle = 0
prev_colour = shape.colour
prev_end_thickness= shape.end_thickness
shape.render(prev_array, prev_angle, prev_colour, prev_end_thickness, ax_n)

fig.show()
"""

"""

fig, ax_n = plt.subplots(figsize=(3, 3))
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
"""
fig, ax_n = plt.subplots(figsize=(3, 3))
line = LineSegment(SegmentType.LINE, (50,100), StartMode.JUMP, 70, 0, 2, 1, 'red', 0.7, True, 0.4, 0, 0)
line2 = LineSegment(SegmentType.LINE, eye_corner_start, StartMode.CONNECT_MID, 30, 90, 1, 2, 'blue', 0, False, 0.5, 0.5, 0)
line3 = LineSegment(SegmentType.LINE, (102,105), StartMode.SPLIT, 50, 90, 2, 4, 'green', 0.3, False, 0.5, 0, 0.5)


design = EyelinerDesign(line)
line.add_child_segment(line2)
line2.add_child_segment(line3)
fig = design.render_design()

#a,b = generate_eyeliner_curve_lines()
#b,c = generate_middle_curve_lines()
fig.show()

negative_score = analyse_negative(design)
print("negative_score:",negative_score)
positive_score = analyse_positive(design)
print("positive_score:",positive_score)
print(is_outside_face_area(line2))


"""
"""
design = EyelinerDesign(random_irregular_polygon())
fig, ax_n = plt.subplots(figsize=(3, 3))
fig = design.render_design(ax_n)
fig.show()
positive_score = analyse_positive(design)
negative_score = analyse_negative(design)
print("analyse_negative score:", negative_score)
print("Positive Score:", positive_score)
"""

