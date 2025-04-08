import copy
import random

import numpy as np
from matplotlib import pyplot as plt

from A import min_negative_score, SegmentType, StartMode, eye_corner_start, random_normal_within_range
from AnalyseDesign import analyse_negative, analyse_positive
from CompareSegments import compare_segments, random_irregular_polygon
from EyelinerDesign import EyelinerDesign
from Segments import LineSegment


def crossover_designs(designs,try_n=0):
    """
       Performs a crossover between trees by swapping a subtree with each design it is breeding with.
    """
    if try_n<20:
        offspring_choice = designs[0]
        if len(offspring_choice.get_all_nodes()) == 1:
            valid_designs = [d for d in designs if len(d.get_all_nodes()) > 1]
            if valid_designs:
                offspring_choice = random.choice(valid_designs)
            else:
                offspring_choice = None
    else:
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
                len_gene_to_breed_nodes = len(gene_to_breed_nodes)
                len_offspring_nodes = len(offspring_nodes)

                # Randomly select a node from each tree (excluding the root to avoid swapping entire trees)
                if len_gene_to_breed_nodes>1 and len_offspring_nodes>1:
                    if len_offspring_nodes ==2:
                        offspring_index = 1
                    else:
                        offspring_index = random.randrange(1, len_offspring_nodes-1)
                    offspring_node = offspring_nodes[offspring_index]

                    if len_gene_to_breed_nodes ==2:
                        gene_to_breed_index = 1
                    else:
                        #Scale that index to the gene_nodes list
                        scaled_index = int(round(offspring_index / len_offspring_nodes * len_gene_to_breed_nodes))
                        if scaled_index >= len_gene_to_breed_nodes:
                            scaled_index = len_gene_to_breed_nodes - 1

                        stddev = max(1,int(len_gene_to_breed_nodes/4))
                        gene_to_breed_index = int(random_normal_within_range(scaled_index,stddev,(1,len_gene_to_breed_nodes-1)))
                    gene_to_breed_node = gene_to_breed_nodes[gene_to_breed_index] # Exclude the root of tree2

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
        return designs[0]

    return offspring

def produce_correct_crossover(designs):
    try_n = 0
    #for design in designs:
    #    overlap_score = analyse_negative(design)
    #    print(f"original overlap_score =={overlap_score}")
    new_design = crossover_designs(designs, try_n)
    new_design.render_design(show=False)
    overlap_score = analyse_negative(new_design)
    #print(f"overlap_score =={overlap_score}")
    while overlap_score < min_negative_score :
        try_n +=1
        new_design = crossover_designs(designs,try_n)
        new_design.render_design(show=False)
        overlap_score = analyse_negative(new_design)
        #print(f"overlap_score =={overlap_score}")

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


def generate_sufficiently_different_gene(old_gene, new_gene_pool, mutation_rate, diff_threshold=0.5, max_attempts=100):
    """
    Generates a mutated gene that is at least `diff_threshold` different from both the parent gene
    and all genes in new_gene_pool.

    Keep mutating gene until it differs enough or max_attempts is reached.
    """
    attempts = 0
    new_gene = old_gene.mutate_design_positive_check(mutation_rate)

    while attempts < max_attempts:
        # Check difference with parent gene
        diff_from_parent = compare_designs(old_gene, new_gene)
        if diff_from_parent < diff_threshold:
            new_gene = new_gene.mutate_design_positive_check(mutation_rate)
            attempts += 1
            continue

        # Check difference with every gene in the new gene pool
        differences = [compare_designs(gene, new_gene) for gene in new_gene_pool]
        if differences and min(differences) < diff_threshold:
            new_gene = new_gene.mutate_design_positive_check(mutation_rate)
            attempts += 1
            continue

        # Candidate is sufficiently different from both the parent and the pool
        return new_gene

    return new_gene

def generate_sufficiently_different_positive_gene_multiple_parents(parents, new_gene_pool, i, mutation_rate,  diff_threshold=0.75, max_attempts=12):
    """
    Generates a mutated gene that is at least `diff_threshold` different from all parent genes
    and all genes in new_gene_pool.

    Keep mutating gene until it differs enough or max_attempts is reached.
    """
    attempts = 0
    # Randomly pick some genes to breed
    num_to_select = random.randint(2, len(parents))
    base = parents[i% len(parents)]
    to_breed = [base]
    available = [p for p in parents if p != base]
    additional = random.sample(available, num_to_select - 1)
    to_breed.extend(additional)
    new_design = produce_correct_crossover(to_breed)
    new_mutated_design = new_design.mutate_design_positive_check(mutation_rate)

    while attempts < max_attempts:
        # Check difference with every parent gene
        differences = [compare_designs(parent, new_mutated_design) for parent in to_breed]
        min_difference = min(differences)
        if min_difference < diff_threshold:
            new_mutated_design = new_mutated_design.mutate_design_positive_check(mutation_rate)
            attempts += 1
            print("parent differences and min(differences) < diff_threshold")
            continue

        # Check difference with every gene in the new gene pool
        differences = [compare_designs(gene, new_mutated_design) for gene in new_gene_pool]
        if differences and min(differences) < diff_threshold:
            new_mutated_design = new_mutated_design.mutate_design_positive_check(mutation_rate)
            attempts += 1
            print("new gene pool differences and min(differences) < diff_threshold")
            continue

        # Candidate is sufficiently different from both the parent and the pool
        return new_mutated_design

    return new_mutated_design

def generate_gene_multiple_parents(parents, index, mutation_rate):
    """
    Generates a mutated gene that is at least `diff_threshold` different from all parent genes
    and all genes in new_gene_pool.

    Keep mutating gene until it differs enough or max_attempts is reached.
    """
    # Randomly pick some genes to breed
    num_to_select = random.randint(2, len(parents))
    base = parents[index]
    to_breed = [base]
    available = [p for p in parents if p != base]
    additional = random.sample(available, num_to_select - 1)
    to_breed.extend(additional)
    new_design = produce_correct_crossover(to_breed)
    new_mutated_design = new_design.mutate_design(mutation_rate, delete=False)

    return new_mutated_design

def breed_new_designs(selected_genes, mutation_rate):
    new_gene_pool=[]
    n_selected = len(selected_genes)
    if n_selected == 1:
        old_gene = selected_genes[0]
        for i in range(6):
            new_design = generate_sufficiently_different_gene(old_gene, new_gene_pool, mutation_rate)

            new_gene_pool.append(new_design)
    else:
        for i in range(6):
            new_design = generate_sufficiently_different_positive_gene_multiple_parents(selected_genes, new_gene_pool, i, mutation_rate)

            new_gene_pool.append(new_design)

    return new_gene_pool

def breed_new_designs_with_auto_selection(selected_genes, mutation_rate, aesthetic_weight=0.7, diversity_weight=0.15, min_diversity_threshold =2):
    new_gene_pool=[]
    n_selected = len(selected_genes)
    if n_selected == 1:
        parent = selected_genes[0]
        for i in range(30):
            new_design = parent.mutate_design(mutation_rate, delete=False)
            new_gene_pool.append(new_design)

        # Score each gene: using analyse_positive for aesthetics,
        # and compare_designs (difference from the parent) for diversity.
        scored_genes = []
        for gene in new_gene_pool:
            aesthetic_score = analyse_positive(gene)
            div_score = compare_designs(gene, parent)
            total_score = (aesthetic_weight * aesthetic_score) + (diversity_weight * div_score)
            scored_genes.append((gene, total_score, aesthetic_score, div_score))

        # Sort the mutations by total score (higher is better)
        scored_genes.sort(key=lambda x: x[1], reverse=True)

        # Try to select the top 3 genes that are sufficiently diverse from each other.
        diversity_threshold = min_diversity_threshold
        new_selected_genes = []
        max_iterations = 10  # safety limit to prevent an infinite loop
        iteration = 0

        while iteration < max_iterations:
            new_selected_genes = []
            if scored_genes:
                # Always take the best-scoring gene
                new_selected_genes.append(scored_genes[0][0])
                # Iterate over the remaining candidates
                for gene, total_score, aesthetic_score, div_score in scored_genes[1:]:
                    # Accept this gene only if it is sufficiently different from all already selected ones
                    if all(compare_designs(gene, sel_gene) >= diversity_threshold for sel_gene in
                           new_selected_genes):
                        new_selected_genes.append(gene)
                    if len(new_selected_genes) >= 3:
                        break
            if len(new_selected_genes) >= 3:
                break  # got enough genes
            else:
                # Relax the diversity requirement and try again
                diversity_threshold *= 0.9  # reduce threshold by 10%
                iteration += 1
    else:
        batch_size = 30 // n_selected
        new_selected_genes = []

        for idx, parent in enumerate(selected_genes):
            batch = []
            for i in range(batch_size):
                # For each batch, call generate_gene_multiple_parents with the current parent's index.
                new_design = generate_gene_multiple_parents(selected_genes, idx, mutation_rate)
                batch.append(new_design)
                #print("batch.append(new_design)")
            # Score each gene in the batch relative to all selected parents.
            scored_batch = []
            for gene in batch:
                aesthetic_score = analyse_positive(gene)
                # Measure diversity as the average difference from all parents.
                div_score = sum(compare_designs(gene, p) for p in selected_genes) / len(selected_genes)
                total_score = (aesthetic_weight * aesthetic_score) + (diversity_weight * div_score)
                scored_batch.append((gene, total_score, aesthetic_score, div_score))
            # Sort the batch by total score (descending) and select the best gene from the batch.
            scored_batch.sort(key=lambda x: x[1], reverse=True)
            if scored_batch:
                new_selected_genes.append(scored_batch[0][0])
                if len(new_selected_genes) < 4:
                    new_selected_genes.append(scored_batch[1][0])

    second_new_gene_pool = breed_new_designs(new_selected_genes, mutation_rate)

    return second_new_gene_pool

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