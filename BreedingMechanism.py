import copy
import random

from A import min_fitness_score
from AnalyseDesign import analyse_negative


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
        return copy.deepcopy(random.choice(designs))

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

def breed_new_designs(selected_genes, mutation_rate):
    new_gene_pool=[]
    n_selected = len(selected_genes)
    if n_selected == 1:
        old_gene = selected_genes[0]
        for i in range(6):
            new_design = old_gene.mutate_design(mutation_rate)
            # print("first overlap_score: ", overlap_score)
            #while analyse_negative(new_design) <= min_fitness_score:
            #    new_design = old_gene.mutate(mutation_rate)
            new_gene_pool.append(new_design)

    else:
        for i in range(6):
            #Randomly pick some genes to breed
            num_to_select = random.randint(2, len(selected_genes))
            to_breed = random.sample(selected_genes, num_to_select)
            new_design = produce_correct_crossover(to_breed)
            print("# crossed")
            new_design = new_design.mutate_design(mutation_rate)
            new_gene_pool.append(new_design)

    return new_gene_pool
