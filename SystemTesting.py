import random
import pytest
import customtkinter as ctk

from A import average_diff_threshold, min_validity_score
from AnalyseDesign import calculate_validity_score
from RandomGene import random_gene
from BreedingMechanism import avg_difference
from DesignPage import DesignPage

class DummyController:
    def __init__(self):
        self.all_saved_simulation_runs = []
    def show_page(self, page_name):
        pass

@pytest.fixture
def dummy_design_page():
    root = ctk.CTk()
    controller = DummyController()
    design_page = DesignPage(parent=root, controller=controller)
    yield design_page
    root.destroy()

@pytest.mark.parametrize("gene_n, num_selected, mutation_rate", [
    (10, 2, 0.6),
    (100, 3, 1.2),
    (170, 1, 1.0),
])
def test_submit_selection_diversity_and_validity(dummy_design_page, gene_n, num_selected, mutation_rate):
    """
    System-level test verifying:
      - After submit_selection() is called,
      - The new gene pool is updated by breeding from the selected genes,
      - Each new design is:
          * Sufficiently diverse from selected designs and each other (avg_difference > average_diff_threshold)
          * Valid (validity score >= min_validity_score)
    """
    pool_size = 6
    system = dummy_design_page
    system.mutation_rate = mutation_rate

    # Populate the gene pool
    system.current_gene_pool = [random_gene(gene_n) for _ in range(pool_size)]
    system.current_gene_pool_figures = []
    system.saved_gene_widgets = {}

    # Simulate user selection
    selected_indices = random.sample(range(pool_size), num_selected)
    system.selected_gene_indices = selected_indices.copy()
    selected_genes = [system.current_gene_pool[i] for i in selected_indices]

    # Generate new designs
    system.submit_selection()
    new_gene_pool = system.current_gene_pool

    assert new_gene_pool, "New gene pool is empty after submit_selection."

    # --- Check DIVERSITY from selected designs ---
    for new_gene in new_gene_pool:
        for orig_gene in selected_genes:
            diff = avg_difference(new_gene, orig_gene)
            assert diff > average_diff_threshold, (
                f"New gene {new_gene} is too similar to selected gene {orig_gene}. "
                f"Avg diff = {diff}, threshold = {average_diff_threshold}"
            )

    # --- Check DIVERSITY between new designs ---
    for i in range(len(new_gene_pool)):
        for j in range(i + 1, len(new_gene_pool)):
            diff = avg_difference(new_gene_pool[i], new_gene_pool[j])
            assert diff > average_diff_threshold, (
                f"New gene {new_gene_pool[i]} is too similar to {new_gene_pool[j]}. "
                f"Avg diff = {diff}, threshold = {average_diff_threshold}"
            )

    # --- Check VALIDITY of each new design ---
    for design in new_gene_pool:
        score = calculate_validity_score(design)
        assert score >= min_validity_score, (
            f"Design {design} has validity score {score} below threshold {min_validity_score}."
        )
