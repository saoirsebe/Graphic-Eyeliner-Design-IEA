import copy
import random

import numpy as np
from matplotlib import pyplot as plt

from A import min_fitness_score, SegmentType, StartMode, eye_corner_start
from AnalyseDesign import analyse_negative
from CompareSegments import compare_segments, random_irregular_polygon
from EyelinerDesign import EyelinerDesign
from Segments import LineSegment


def crossover_designs(designs):
    """
       Performs a crossover between trees by swapping a subtree with each design it is breeding with.
    """
    valid_designs = [d for d in designs if len(d.get_all_nodes()) > 1]
    if valid_designs:
        offspring_choice = random.choice(valid_designs)
    else:
        offspring_choice = None

    if offspring_choice:
        offspring = copy.deepcopy(offspring_choice)
        for design in designs:
            if design != offspring:
                gene_to_breed = copy.deepcopy(design)
                # Get all nodes in both trees
                offspring_nodes = offspring.get_all_nodes()
                gene_to_breed_nodes = gene_to_breed.get_all_nodes()

                # Randomly select a node from each tree (excluding the root to avoid swapping entire trees)
                if len(gene_to_breed_nodes)>1 and len(offspring_nodes)>1:
                    offspring_node = random.choice(offspring_nodes[1:])  # Exclude the root of tree1
                    gene_to_breed_node = random.choice(gene_to_breed_nodes[1:])  # Exclude the root of tree2

                    # Find parents of the selected nodes
                    offspring_parent = offspring.find_the_parent(offspring.root, offspring_node)
                   # print("offspring_parent children before:", offspring_parent.children)

                    # Swap subtree with offspring
                    if offspring_parent:
                        # Remove the nodes from their current parents
                        offspring_parent.remove_child_segment(offspring_node)
                        offspring_parent.add_child_segment(gene_to_breed_node)
                        #print("offspring_parent children after:", offspring_parent.children)
                    else:
                        print("no offspring_parent")
    else:
        return random.choice(designs)

    return offspring

def produce_correct_crossover(designs):
    new_design = crossover_designs(designs)
    new_design.render_design(show=False)
    overlap_score = analyse_negative(new_design)
    while overlap_score <= min_fitness_score:
        new_design = crossover_designs(designs)
        new_design.render_design(show=False)
        overlap_score = analyse_negative(new_design)

    return new_design


def compare_designs(parent, new_design):
    difference = 0
    parent_nodes = parent.get_all_nodes()
    new_design_nodes = new_design.get_all_nodes()
    min_len = min(len(parent_nodes), len(new_design_nodes))

    # Compare corresponding segments:
    for i in range(min_len):
        difference += compare_segments(parent_nodes[i], new_design_nodes[i])
    # Penalize for extra nodes in one design:
    difference += abs(len(parent_nodes) - len(new_design_nodes)) * 10
    return difference



def breed_new_designs(selected_genes, mutation_rate):
    new_gene_pool=[]
    n_selected = len(selected_genes)
    if n_selected == 1:
        old_gene = selected_genes[0]
        for i in range(6):
            new_design = old_gene.mutate_design(mutation_rate)
            difference = compare_designs(old_gene, new_design)
            while difference < 0.5:
                new_design = new_design.mutate_design(mutation_rate)
                difference = compare_designs(old_gene, new_design)
            new_gene_pool.append(new_design)

    else:
        for i in range(6):
            #Randomly pick some genes to breed
            num_to_select = random.randint(2, len(selected_genes))
            to_breed = random.sample(selected_genes, num_to_select)
            new_design = produce_correct_crossover(to_breed)
            try_unique = 0
            while new_design in selected_genes and try_unique<4:
                new_design = produce_correct_crossover(to_breed)
                try_unique+=1

            new_mutated_design = new_design.mutate_design(mutation_rate)
            difference = compare_designs(new_design, new_mutated_design)
            while difference < 0.5:
                new_design = new_design.mutate_design(mutation_rate)
                difference = compare_designs(new_design, new_mutated_design)

            new_gene_pool.append(new_design)

    return new_gene_pool
"""
fig, ax_n = plt.subplots(figsize=(3, 3))
line = LineSegment(SegmentType.LINE, (50,100), StartMode.JUMP, 70, 0, 2, 1, 'red', 0.7, True, 0.4, 0, 0)
line2 = LineSegment(SegmentType.LINE, eye_corner_start, StartMode.CONNECT_MID, 30, 90, 1, 2, 'blue', 0, False, 0.5, 0.5, 0)
line3 = LineSegment(SegmentType.LINE, (102,105), StartMode.SPLIT, 50, 90, 2, 4, 'green', 0.3, False, 0.5, 0, 0.5)


design = EyelinerDesign(line)
line.add_child_segment(line2)
line2.add_child_segment(line3)
fig = design.render_design()

fig.show()
design2 = copy.deepcopy(design)
design2 = design2.mutate_design()
design2 = design2.mutate_design()
design2 = design2.mutate_design()
#design2 = design2.mutate_design()

#design2.root.start = (design2.root.start[0] + 100, design2.root.start[1])
fig, ax_n = plt.subplots(figsize=(3, 3))
fig = design2.render_design()

fig.show()

difference = compare_designs(design, design2)
print("difference =",difference)
"""