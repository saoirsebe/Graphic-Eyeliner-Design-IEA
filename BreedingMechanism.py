import copy
import multiprocessing
import random
from io import BytesIO
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
from matplotlib import pyplot as plt

from A import min_validity_score, SegmentType, StartMode, eye_corner_start, random_normal_within_range, diff_threshold, \
    average_diff_threshold, Second_population_size, first_population_size, max_crossover_attempts, max_mutation_attempts
from AnalyseDesign import calculate_validity_score, calculate_aesthetic_fitness_score
from CompareSegments import compare_segments, random_irregular_polygon
from EyelinerDesign import EyelinerDesign
from Segments import LineSegment


def crossover_designs(designs,try_n=0):
    """
       Performs a crossover between trees by swapping a subtree with each design it is breeding with.
    """
    if try_n<max_crossover_attempts:
        offspring_choice = designs[0]
        if len(offspring_choice.get_all_nodes()) == 1:
            valid_designs = [d for d in designs if len(d.get_all_nodes()) > 1]
            if valid_designs:
                offspring_choice = random.choice(valid_designs)
            else:
                offspring_choice = None
    else:
        #print("aaaaaaaaa base not used")
        return designs[0]

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
    validity_score = calculate_validity_score(new_design)
    #print(f"overlap_score =={overlap_score}")
    while validity_score < min_validity_score :
        try_n +=1
        new_design = crossover_designs(designs,try_n)
        new_design.render_design(show=False)
        validity_score = calculate_validity_score(new_design)
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


def figure_to_array(fig):
    """
    Convert a matplotlib figure to a NumPy array.

    Parameters:
        fig: The matplotlib figure to convert.
        convert_gray: If True, convert the image to grayscale ('L' mode in PIL).

    Returns:
        image_array: NumPy array of the rendered figure.
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    im = Image.open(buf)
    image_array = np.array(im)
    buf.close()
    return image_array

def compare_design_images(design2, design1, image1=None):
    fig2 = design2.render_design()
    image2 = figure_to_array(fig2)
    plt.close(fig2)
    if image1 is None:
        fig1 = design1.render_design()
        image1 = figure_to_array(fig1)
        plt.close(fig1)

    genetic_difference = compare_designs(design1,design2)
    if image1.shape == image2.shape:
        image2_gray = np.dot(image2[..., :3], [0.2989, 0.5870, 0.1140])
        image1_gray = np.dot(image1[..., :3], [0.2989, 0.5870, 0.1140])
        err = np.mean((image1_gray.astype("float") - image2_gray.astype("float")) ** 2)
        return err, genetic_difference * 20
    else:
        return genetic_difference * 20, genetic_difference * 20


def generate_sufficiently_different_gene(old_gene, old_image, new_gene_pool, mutation_rate):
    """
    Generates a mutated gene that is at least `diff_threshold` different from both the parent gene
    and all genes in new_gene_pool.
    """
    gene_attempts = []
    attempts = 0
    new_gene = old_gene.mutate_design(mutation_rate)


    while attempts < max_mutation_attempts:
        img_diff, gen_diff = compare_design_images(new_gene, old_gene, old_image)
        if img_diff < average_diff_threshold or gen_diff < diff_threshold:
            gene_attempts.append(new_gene)
            new_gene = old_gene.mutate_design(mutation_rate)
            attempts += 1
            continue

        differences = [avg_difference(new_gene, gene) for gene in new_gene_pool]
        if differences and any(avr_diff < average_diff_threshold for (avr_diff) in differences):
            new_gene = old_gene.mutate_design(mutation_rate)
            attempts += 1
            continue

        return new_gene

    print("no mutated diverse enough")
    # If no gene was sufficiently different during attempts, pick the one that differs the most.
    # Calculates an overall difference score:
    def overall_difference(gene):
        # Differences compared to the parent
        img_diff, gen_diff = compare_design_images(gene, old_gene, old_image)
        parent_diff = img_diff + gen_diff
        # Sum of differences compared to each gene in the new gene pool
        pool_diff = sum(avg_difference(gene, candidate) for candidate in new_gene_pool)
        return parent_diff + pool_diff

    if gene_attempts:
        best_gene = max(gene_attempts, key=overall_difference)
        return best_gene.mutate_design(mutation_rate)

    return new_gene


def generate_sufficiently_different_gene_multiple_parents(parents, new_gene_pool, i, mutation_rate,
                                                          max_attempts=20, old_selected_genes=None):
    """
    Generates a mutated gene that is at least `diff_threshold` different from all parent genes
    and all genes in new_gene_pool. It continues mutating until such a gene is found, or until
    max_attempts is reached. If no candidate meets the criteria, the most diverse candidate is chosen.
    """
    attempts = 0
    gene_attempts = []  # List to store each mutated candidate that doesn't meet the threshold

    # Randomly select a subset of parents to breed
    num_to_select = random.randint(2, len(parents))
    base = parents[i % len(parents)]
    to_breed = [base]
    available = [p for p in parents if p != base]
    additional = random.sample(available, num_to_select - 1)
    to_breed.extend(additional)

    new_design = produce_correct_crossover(to_breed)
    new_mutated_design = new_design.mutate_design(mutation_rate)

    # Determine which parent set to compare against and define the current difference threshold
    parents_to_compare = old_selected_genes if old_selected_genes else parents
    current_diff_threshold = average_diff_threshold if old_selected_genes else average_diff_threshold / 2

    #print(parents)
    #print(parents_to_compare)
    while attempts < max_attempts:
        # Check differences with each parent
        parent_diffs = [avg_difference(parent, new_mutated_design) for parent in parents_to_compare]
        if any(avr_diff < current_diff_threshold for avr_diff in parent_diffs):
            gene_attempts.append(new_mutated_design)
            new_mutated_design = new_design.mutate_design(mutation_rate)
            attempts += 1
            continue

        # Check differences with each gene in the new gene pool
        pool_diffs = [avg_difference(gene, new_mutated_design) for gene in new_gene_pool]
        if pool_diffs and any(avr_diff < average_diff_threshold for avr_diff in pool_diffs):
            gene_attempts.append(new_mutated_design)
            new_mutated_design = new_design.mutate_design(mutation_rate)
            attempts += 1
            continue

        # If the candidate is sufficiently different from both the parents and the pool, return it.
        return new_mutated_design

    print("None mutated sufficiently diverse")
    # If we reach here, no candidate met the threshold;
    # select the candidate that has the highest overall difference.
    if gene_attempts:
        def overall_difference(candidate):
            parent_sum = sum(avg_difference(parent, candidate) for parent in parents_to_compare)
            pool_sum = sum(avg_difference(gene, candidate) for gene in new_gene_pool) if new_gene_pool else 0
            return parent_sum + pool_sum

        best_candidate = max(gene_attempts, key=overall_difference)
        return best_candidate.mutate_design(mutation_rate)

    # As a fallback, return the last mutation attempt.
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

def breed_new_designs(selected_genes, mutation_rate, old_selected_genes = None):
    new_gene_pool=[]
    n_selected = len(selected_genes)
    if n_selected == 1:
        old_gene = selected_genes[0]
        fig1 = old_gene.render_design()
        old_image = figure_to_array(fig1)
        plt.close(fig1)
        for i in range(Second_population_size):
            new_design = generate_sufficiently_different_gene(old_gene, old_image, new_gene_pool, mutation_rate)

            new_gene_pool.append(new_design)

        new_gene_pool_scored = []
        for gene in new_gene_pool:
            aesthetic_score = calculate_aesthetic_fitness_score(gene)
            new_gene_pool_scored.append((gene, aesthetic_score))

        # Sort the mutations by aesthetic score and show top 6
        new_gene_pool_scored.sort(key=lambda x: x[1], reverse=True)
        gene_pool_to_show = [gene for gene, score in new_gene_pool_scored[:6]]
    else:
        batch_size = Second_population_size // n_selected
        batch_candidates = []

        for idx, parent in enumerate(selected_genes):
            for _ in range(batch_size):
                new_design = generate_sufficiently_different_gene_multiple_parents(selected_genes, new_gene_pool, idx, mutation_rate, old_selected_genes=old_selected_genes)

                batch_candidates.append((new_design, idx))
                new_gene_pool.append(new_design)

        new_gene_pool_scored = []
        for gene, idx in batch_candidates:
            aesthetic_score = calculate_aesthetic_fitness_score(gene)
            new_gene_pool_scored.append((gene, idx, aesthetic_score))

        # Sort the mutations by aesthetic score and show top 6
        new_gene_pool_scored.sort(key=lambda x: x[2], reverse=True)

        # Group candidates by parent's index, ensuring that each batch contributes its best candidate.
        from collections import defaultdict

        grouped = defaultdict(list)
        for gene, idx, score in new_gene_pool_scored:
            grouped[idx].append((gene, score))

        mandatory_candidates = []
        for idx in grouped:
            # Select the best candidate from this parent's batch.
            best_gene, best_score = max(grouped[idx], key=lambda x: x[1])
            mandatory_candidates.append(best_gene)

        # Ensure at least one candidate from each parent's batch appears in gene_pool_to_show.
        # Then fill list (up to 6) with the overall top genes.
        gene_pool_to_show = list(mandatory_candidates)  # start with the best from each batch
        selected_set = set(gene_pool_to_show)

        # Add additional candidates from the overall sorted list if we don't have 6 yet.
        for gene, idx, score in new_gene_pool_scored:
            if gene not in selected_set:
                gene_pool_to_show.append(gene)
                selected_set.add(gene)
            if len(gene_pool_to_show) >= 6:
                break

        # Limit the list to 6.
        gene_pool_to_show = gene_pool_to_show[:6]

    return gene_pool_to_show


def avg_difference(design1, design2):
    img, gen = compare_design_images(design1, design2)
    return (img + gen) / 2


def auto_selection_one_parent(selected_genes, parent, mutation_rate):
    fig1 = parent.render_design()
    old_image = figure_to_array(fig1)
    plt.close(fig1)
    with multiprocessing.Pool() as pool:
        # Create a list of arguments for each call.
        args = [(parent,old_image, [], mutation_rate) for _ in range(first_population_size)]
        new_gene_pool = pool.starmap(generate_sufficiently_different_gene, args)

    # Score each gene: using analyse_positive for aesthetics,
    # and compare_design_images (difference from the parent) for diversity.
    scored_genes = []
    for gene in new_gene_pool:
        aesthetic_score = calculate_aesthetic_fitness_score(gene)
        scored_genes.append((gene, aesthetic_score))

    # Sort the mutations by aesthetic score
    scored_genes.sort(key=lambda x: x[1], reverse=True)

    # Initial threshold for checking pairwise diversity among new selected designs
    selected_threshold = average_diff_threshold / 1.5
    max_iterations = 10
    iteration = 0
    new_selected_genes = []

    while iteration < max_iterations:
        for gene, aest in scored_genes:
            if not new_selected_genes:
                # Always add the highest aesthetic design as a starting point.
                new_selected_genes.append(gene)
            else:
                if gene not in new_selected_genes:
                    # Add this gene only if it is sufficiently different to all already selected ones.
                    if all(avg_difference(gene, sel) >= selected_threshold for sel in new_selected_genes):
                        new_selected_genes.append(gene)
            if len(new_selected_genes) >= 6:
                # We aim for a maximum of six designs.
                break
        if len(new_selected_genes) >= 4:
            # Sufficient number selected.
            break
        else:
            # Relax the selected diversity requirement and try again.
            selected_threshold *= 0.9
            iteration += 1


        # Return between 4 and 6 selected designs.
    return new_selected_genes[:6]


def auto_selection_multiple_parents(selected_genes, batch_size, mutation_rate):
    # 1. Generate candidate pool for each parent.
    if len(selected_genes) ==2:
        max_batch_picked = 3
    elif len(selected_genes) ==3:
        max_batch_picked = 2
    else:
        max_batch_picked = 1

    new_selected_genes = []
    for idx, parent in enumerate(selected_genes):
        with multiprocessing.Pool() as pool:
            args = [(selected_genes, [], idx, mutation_rate) for _ in range(batch_size)]
            batch_candidates = pool.starmap(generate_sufficiently_different_gene_multiple_parents, args)

        # 2. Score each candidate based on aesthetics
        scored_candidates = [(gene, calculate_aesthetic_fitness_score(gene)) for gene in batch_candidates]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)


        # 4. Select between 4 and 6 mutually diverse candidates using iterative threshold relaxation.
        iteration = 0
        max_iterations = 20
        # Initial selected threshold for mutual diversity.
        selected_threshold = average_diff_threshold / 1.5
        batch_selected_genes = []  # reset the selection for each iteration

        while iteration < max_iterations:
            for gene, aest in scored_candidates:
                # Add the candidate only if it is sufficiently different from every
                # already selected candidate. Use the helper avg_difference.
                if all(avg_difference(gene, sel) >= selected_threshold for sel in new_selected_genes) and all(avg_difference(gene, sel) >= selected_threshold for sel in batch_selected_genes):
                    batch_selected_genes.append(gene)
                # Stop early if we have reached our maximum selected for one batch.
                if len(batch_selected_genes) >= max_batch_picked:
                    break
            # If we have at least 1 candidate, selection is successful.
            if len(batch_selected_genes) >= 1:
                break
            else:
                # Otherwise, relax the pairwise threshold by 10% and try again.
                selected_threshold *= 0.9
                iteration += 1

        if len(batch_selected_genes)==0:
            batch_selected_genes.append(scored_candidates[0])

        new_selected_genes.extend(batch_selected_genes)
    # Return between 4 and 6 selected designs.
    return new_selected_genes[:6]

def auto_selection_multiple_parents_old(selected_genes, batch_size, mutation_rate):
    new_selected_genes = []

    for idx, parent in enumerate(selected_genes):
        batch = []
        for i in range(batch_size):
            # For each batch, call generate_gene_multiple_parents with the current parent's index.
            new_design = generate_gene_multiple_parents(selected_genes, idx, mutation_rate)
            batch.append(new_design)

        # Score each gene in the batch relative to all selected parents.
        scored_batch = []
        for gene in batch:
            aesthetic_score = calculate_aesthetic_fitness_score(gene)
            # Measure diversity as the average difference from all parents.
            div_scores = [(img + gen) / 2 for (img, gen) in (compare_design_images(gene, p) for p in selected_genes)]
            div_score = sum(div_scores) / len(div_scores)
            scored_batch.append((gene, aesthetic_score, div_score))
        # Sort the batch by total score (descending) and select the best gene from the batch.
        scored_batch.sort(key=lambda x: x[1], reverse=True)
        if scored_batch:
            new_selected_genes.append(scored_batch[0][0])
            new_selected_genes.append(scored_batch[1][0])
    if len(new_selected_genes) < 6:
        new_selected_genes.append(scored_batch[2][0])

    return new_selected_genes


def breed_new_designs_with_auto_selection(selected_genes, mutation_rate, aesthetic_weight=0.7, diversity_weight=0.15):
    n_selected = len(selected_genes)
    if n_selected == 1:
        parent = selected_genes[0]
        new_to_breed = auto_selection_one_parent(selected_genes, parent, mutation_rate)
    else:
        batch_size = first_population_size // n_selected
        new_to_breed = auto_selection_multiple_parents(selected_genes, batch_size, mutation_rate)

    for design in new_to_breed:
        fig = design.render_design()
        fig.show()

    second_genes_to_show = breed_new_designs(new_to_breed, mutation_rate, selected_genes)

    return second_genes_to_show

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

difference = compare_design_images(design, design2)
print("difference =",difference)
"""