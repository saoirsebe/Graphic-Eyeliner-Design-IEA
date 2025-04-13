from graphviz import Digraph
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO


def create_flowchart():
    dot = Digraph(comment="IEA Flowchart with Loop", format="png")

    # Define nodes (rectangle for process, diamond for decisions)
    dot.node("start", "User Selects Designs", shape="rectangle")
    dot.node("dec1", "More than 1 Design Selected?", shape="diamond")
    dot.node("crossover", "Apply Crossover Mechanism\n(Negative Score >= Minimum)", shape="rectangle")
    dot.node("mutation", "Apply Mutation\n(Pos Score doesn't decrease & Neg Score >= Min)", shape="rectangle")
    dot.node("diversity", "Different Enough from Parents?", shape="diamond")
    dot.node("add", "Add Design to New Gene Pool", shape="rectangle")
    dot.node("loop", "i < 6?", shape="diamond")
    dot.node("end", "Process Complete", shape="rectangle")

    # Create edges with labels for decisions:
    dot.edge("start", "dec1")
    dot.edge("dec1", "crossover", label="Yes")
    dot.edge("dec1", "mutation", label="No")
    dot.edge("crossover", "mutation")
    dot.edge("mutation", "diversity")
    dot.edge("diversity", "add", label="Yes")
    dot.edge("diversity", "mutation", label="No")
    dot.edge("add", "loop")
    dot.edge("loop", "crossover", label="Yes\n(Repeat)", _attributes={'dir': 'back'})
    dot.edge("loop", "end", label="No\n(6 Designs Added)")

    return dot


def create_for_item_loop_flowchart():
    dot = Digraph(comment="For Each Item in List Loop", format="png")

    # Nodes:
    dot.node("start", "Start Loop:\nfor item in list", shape="rectangle")
    dot.node("process", "Process Current Item", shape="rectangle")
    dot.node("decision", "More Items Remain?", shape="diamond")
    dot.node("end", "End Loop", shape="rectangle")

    # Edges:
    dot.edge("start", "process")
    dot.edge("process", "decision")
    dot.edge("decision", "process", label="Yes")
    dot.edge("decision", "end", label="No")

    return dot


def create_random_gene_node_flowchart():
    dot = Digraph(comment="Flowchart for random_gene_node")
    dot.graph_attr.update(dpi='500')
    # Nodes for random_gene_node flowchart
    dot.node("A", "Start random_gene_node")
    dot.node("B", "Create new_node via random_segment(prev_colour)")
    dot.node("C", "Set previous thickness array & previous array from parent\nSet parent's absolute angle as prev_angle")
    dot.node("D", "Render new_node with prev_array, prev_angle, prev_colour,\nprev_end_thickness_array")
    dot.node("E", "Initialize regen_count = 0")
    dot.node("F", "Calculate new_segment_score = check_new_segments_negative_score(design, new_node)")
    dot.node("G", "Is new_segment_score < min_negative_score\nand regen_count < node_re_gen_max?")
    dot.node("H", "Re-generate new_node via random_segment(prev_colour)")
    dot.node("I", "Render new_node again using same previous data")
    dot.node("J", "Increment regen_count")
    dot.node("K", "Update new_segment_score")
    dot.node("L", "Has regen_count reached node_re_gen_max?")
    dot.node("M", "Return False, min_negative_score * 2")
    dot.node("N", "Append new_node to parent's children")
    dot.node("O", "Determine n_of_children = n_of_children_decreasing_likelihood(segment_number,\n depth, max_segments, 1.6, 0.6, number_of_children_range)")
    dot.node("P", "Update prev_colour = new_node.colour\nIncrement depth")
    dot.node("Q", "For each child (loop over n_of_children):")
    dot.node("R", "Increment segment_number")
    dot.node("S", "Recursively call random_gene_node(design, new_node, prev_colour,\nsegment_number, depth)")
    dot.node("T", "Accumulate child score into new_segment_score")
    dot.node("U", "Did recursive call succeed\nand is new_segment_score ≥ min_negative_score?")
    dot.node("V", "Return False, min_negative_score * 2")
    dot.node("W", "Return True, new_segment_score")

    # Edges for random_gene_node flowchart
    dot.edge("A", "B")
    dot.edge("B", "C")
    dot.edge("C", "D")
    dot.edge("D", "E")
    dot.edge("E", "F")
    dot.edge("F", "G")
    dot.edge("G", "N", label="Yes")
    dot.edge("G", "H", label="No")
    dot.edge("H", "I")
    dot.edge("I", "J")
    dot.edge("J", "K")
    dot.edge("K", "G")
    dot.edge("G", "L", label="Regen limit reached")
    dot.edge("L", "M", label="Yes")
    dot.edge("L", "N", label="No")
    dot.edge("N", "O")
    dot.edge("O", "P")
    dot.edge("P", "Q")
    dot.edge("Q", "R")
    dot.edge("R", "S")
    dot.edge("S", "T")
    dot.edge("T", "U")
    dot.edge("U", "V", label="No")
    dot.edge("U", "W", label="Yes")

    return dot

def create_random_gene_flowchart():
    dot = Digraph(comment="Flowchart for random_gene")
    dot.graph_attr.update(dpi='500')
    # Nodes for random_gene flowchart
    dot.node("A", "Start random_gene(gene_n)")
    dot.node("B", "Set success = False")
    dot.node("C", "Enter while loop (until success is True)")
    dot.node("D", "Set success = True\nInitialize root_score = -1")
    dot.node("E", "Enter inner loop while root_score < 0")
    dot.node("F", "Determine design based on gene_n:\n• If gene_n between (initial_gene_pool_size/3, 2/3)\n• If gene_n ≤ 2/3 of initial_gene_pool_size\n• Else: random_segment design")
    dot.node("G", "Set root_node = design.root\nSet prev_colour = root_node.colour\nInitialize segment_number = 0")
    dot.node("H", "Render root_node with its start position, angle = 0, etc.")
    dot.node("I", "Is root_node outside face area?")
    dot.node("J", "Set root_score = 2 * min_negative_score")
    dot.node("K", "Else, set root_score = -is_in_eye(root_node)")
    dot.node("L", "Calculate n_of_children using random_normal_within_range")
    dot.node("M", "Set total_score = root_score")
    dot.node("N", "For each child in n_of_children (loop):")
    dot.node("O", "Increment segment_number")
    dot.node("P", "Call random_gene_node(design, root_node, prev_colour,\nsegment_number, depth=0)")
    dot.node("Q", "Add child_score to total_score")
    dot.node("R", "Did child generation succeed\nand is total_score ≥ min_negative_score?")
    dot.node("S", "If not, set success = False")
    dot.node("T", "Render the complete design (without showing)")
    dot.node("U", "Does analyse_negative(design) ≥ min_negative_score?")
    dot.node("V", "If not, set success = False")
    dot.node("W", "Is success True?")
    dot.node("X", "Return the design")

    # Edges for random_gene flowchart
    dot.edge("A", "B")
    dot.edge("B", "C")
    dot.edge("C", "D")
    dot.edge("D", "E")
    dot.edge("E", "F")
    dot.edge("F", "G")
    dot.edge("G", "H")
    dot.edge("H", "I")
    dot.edge("I", "J", label="Yes")
    dot.edge("I", "K", label="No")
    dot.edge("J", "L")
    dot.edge("K", "L")
    dot.edge("L", "M")
    dot.edge("M", "N")
    dot.edge("N", "O")
    dot.edge("O", "P")
    dot.edge("P", "Q")
    dot.edge("Q", "R")
    dot.edge("R", "S", label="No")
    dot.edge("R", "N", label="Yes")
    dot.edge("S", "T")
    dot.edge("T", "U")
    dot.edge("U", "V", label="No")
    dot.edge("U", "W", label="Yes")
    dot.edge("V", "C")
    dot.edge("W", "X")

    return dot


# Create the flowchart
node_flowchart = create_random_gene_node_flowchart()
gene_flowchart = create_random_gene_flowchart()

# Render and display the node_flowchart in-memory
node_png = node_flowchart.pipe(format="png")
node_img = mpimg.imread(BytesIO(node_png), format="png")
plt.figure(figsize=(12, 8))
plt.imshow(node_img)
plt.title("Flowchart for random_gene_node")
plt.axis("off")
plt.show()

# Render and display the gene_flowchart in-memory
gene_png = gene_flowchart.pipe(format="png")
gene_img = mpimg.imread(BytesIO(gene_png), format="png")
plt.figure(figsize=(12, 8))
plt.imshow(gene_img)
plt.title("Flowchart for random_gene")
plt.axis("off")
plt.show()



