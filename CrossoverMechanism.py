import random

def crossover_designs(design1, design2):
    """
       Performs a crossover between two trees by swapping subtrees.
       """
    # Get all nodes in both trees
    nodes1 = design1.get_all_nodes()
    nodes2 = design2.get_all_nodes()

    # Randomly select a node from each tree (excluding the root to avoid swapping entire trees)
    node1 = random.choice(nodes1[1:])  # Exclude the root of tree1
    node2 = random.choice(nodes2[1:])  # Exclude the root of tree2

    # Find parents of the selected nodes
    parent1 = design1.find_parent(design1.root, node1)
    parent2 = design2.find_parent(design2.root, node2)

    # Swap the subtrees
    if parent1 and parent2:
        # Remove the nodes from their current parents
        parent1.children.remove(node1)
        parent2.children.remove(node2)

        # Swap them
        parent1.children.append(node2)
        parent2.children.append(node1)
