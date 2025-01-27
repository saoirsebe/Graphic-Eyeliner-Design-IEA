import copy
import random
from conda.common.configuration import raise_errors
from AnalyseDesign import analyse_negative
from Segments import *

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
            self.render_node( child, prev_array, prev_angle, prev_colour, prev_end_thickness_array,ax_n)

    def render_design(self,show = True):
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
