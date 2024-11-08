import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import comb
from EAOverallGenome import *
import random



def random_gene(gene_n):
    design = EyelinerDesign()
    n_objects = random.randint(1,10)
    next_start_thickness = random.uniform(1,5)

    #Parameter ranges:
    start_x_range = (-1,10)
    start_y_range = (0,10)
    #start_mode_options = list(StartMode)
    #Segment_type_options = list(SegmentType)
    length_range = (0, 5)
    direction_range = (0, 360)
    thickness_range = (1, 5)
    colour_options = ["red", "green", "blue", "yellow", "cyan", "magenta"]
    curviness_range = (0, 10)
    curve_direction_range = (0, 360)
    curve_location_range = (0,1)
    center_x_range = (-2,2)
    center_y_range = (-2,2)
    radius_range = (0,2)
    arm_length_range = (0,2)
    num_points_range = (3,20)
    asymmetry_range = (0,3)
    #curved_range = ["True","False"]

    for i in range(n_objects):
        if (gene_n ==0 and i==0) or (gene_n ==1 and i==0):
            segment_start = (3,1.5)
        else:
            segment_start = (random.uniform(*start_x_range), random.uniform(*start_y_range))

        # Adding a new segment to the design
        new_segment_type = random.choice(list(SegmentType))
        start_type = random.choice(list(StartMode))
        if new_segment_type == SegmentType.LINE:
            new_segment = create_segment(
                segment_type=new_segment_type,
                start = segment_start,
                start_mode=start_type,
                length=random.uniform(*length_range),
                direction=random.uniform(*direction_range),
                start_thickness=next_start_thickness,
                end_thickness=random.uniform(*thickness_range),
                color=random.choice(colour_options),
                curviness=random.uniform(*curviness_range),
                curve_direction=random.uniform(*curve_direction_range),
                curve_location=random.uniform(*curve_location_range)
            )
        elif new_segment_type == SegmentType.STAR:
            new_segment = create_segment(
                segment_type=new_segment_type,
                start = segment_start,
                start_mode=start_type,
                radius=random.uniform(*radius_range),
                arm_length=random.uniform(*arm_length_range),
                num_points=random.randint(*num_points_range),
                asymmetry=random.uniform(*asymmetry_range),
                curved=random.choice([True, False]),
                end_thickness=random.uniform(*thickness_range),
            )

        design.add_segment(new_segment)
        next_start_thickness = design.get_start_thickness()
    design.update_design_info()
    return design


def initialise_gene_pool():
    gene_pool = [random_gene(i) for i in range(6)]  # Generate 6 random genes
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # Create a 2x3 grid of subplots
    axes = axes.flatten()
    # Render each gene in its corresponding subplot
    for idx, gene in enumerate(gene_pool):
        ax = axes[idx]
        ax.set_title(f"Design {idx + 1}")
        gene.render(ax)  # Render each gene on its specific subplot

    plt.tight_layout()
    plt.show()

#initialise_gene_pool()

design = EyelinerDesign()
next_start_thickness = random.uniform(1,5)

#Parameter ranges:
start_x_range = (-1,10)
start_y_range = (0,10)
#start_mode_options = list(StartMode)
#Segment_type_options = list(SegmentType)
length_range = (0, 5)
direction_range = (0, 360)
thickness_range = (1, 5)
colour_options = ["red", "green", "blue", "yellow", "cyan", "magenta"]
curviness_range = (0, 10)
curve_direction_range = (0, 360)
curve_location_range = (0,1)
center_x_range = (-2,2)
center_y_range = (-2,2)
radius_range = (0,2)
arm_length_range = (0,2)
num_points_range = (3,20)
asymmetry_range = (0,3)

plt.figure(figsize=(6, 6))
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlim(-5,10)
ax.set_ylim(-5,10)
new_segment = create_segment(
                segment_type=SegmentType.LINE,
                start = (0,0),
                start_mode=StartMode.CONNECT,
                length=random.uniform(*length_range),
                relative_angle=random.uniform(*direction_range),
                start_thickness=next_start_thickness,
                end_thickness=random.uniform(*thickness_range),
                color=random.choice(colour_options),
                curviness=random.uniform(*curviness_range),
                curve_direction=random.uniform(*curve_direction_range),
                curve_location=random.uniform(*curve_location_range)
            )
design.add_segment(new_segment)
next_start_thickness = design.get_start_thickness()

new_star_segment = create_segment(
                segment_type=SegmentType.STAR,
                start = (0,0),
                start_mode=StartMode.CONNECT,
                radius=random.uniform(*radius_range),
                arm_length=random.uniform(*arm_length_range),
                num_points=random.randint(*num_points_range),
                asymmetry=random.uniform(*asymmetry_range),
                curved=random.choice([True, False]),
                end_thickness=random.uniform(*thickness_range),
                relative_angle = random.uniform(*direction_range)
            )
design.add_segment(new_star_segment)
next_start_thickness = design.get_start_thickness()

new_line_segment = create_segment(
                segment_type=SegmentType.LINE,
                start = (0,0),
                start_mode=StartMode.CONNECT,
                length=random.uniform(*length_range),
                relative_angle=random.uniform(*direction_range),
                start_thickness=next_start_thickness,
                end_thickness=random.uniform(*thickness_range),
                color=random.choice(colour_options),
                curviness=random.uniform(*curviness_range),
                curve_direction=random.uniform(*curve_direction_range),
                curve_location=random.uniform(*curve_location_range)
            )
design.add_segment(new_line_segment)

design.render(ax)
plt.show()