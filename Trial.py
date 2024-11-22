import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import comb
#from dask_expr.diagnostics import analyze
from EAOverallGenome import *
import random

n_objects = random.randint(4,15)
#Parameter ranges:
start_x_range = (-1,7)
start_y_range = (0,6)
#start_mode_options = list(StartMode)
#Segment_type_options = list(SegmentType)
length_range = (0.5, 5)
direction_range = (0, 360)
thickness_range = (1, 5)
colour_options = [
    "red", "green", "blue", "cyan", "magenta", "black", "orange", "purple",
    "pink", "brown", "gray", "lime", "navy",
    "teal", "maroon", "gold", "olive"
]
curviness_range = (0, 10)
curve_direction_range = (0, 360)
curve_location_range = (0,1)
radius_range = (0,1.5)
arm_length_range = (0,1.5)
num_points_range = (3,15)
asymmetry_range = (0,3)

def random_gene(gene_n):
    design = EyelinerDesign()
    for i in range(n_objects):
        if i==0:#(gene_n ==0 and i==0) or (gene_n ==1 and i==0):
            segment_start = (3, 1.5)
        else:
            segment_start = (random.uniform(*start_x_range), random.uniform(*start_y_range))

        # Adding a new segment to the design
        #new_segment_type = random.choice(list(SegmentType))
        new_segment_type = SegmentType.LINE if random.random() < 0.8 else SegmentType.STAR
        if new_segment_type == SegmentType.LINE:
            start_mode = random.choice(list(StartMode))
            new_segment = create_segment(
                segment_type=SegmentType.LINE,
                start=segment_start,
                start_mode=start_mode,
                length=random.uniform(*length_range),
                relative_angle=random.uniform(*direction_range),
                start_thickness=random.uniform(*thickness_range),
                end_thickness=random.uniform(*thickness_range),
                colour=random.choice(colour_options),
                curviness = 0 if random.random() < 0.5 else random.uniform(*curviness_range),
                curve_direction=random.uniform(*curve_direction_range),
                curve_location=random.uniform(*curve_location_range),
                start_location=random.uniform(*curve_location_range),
                split_point = random.uniform(*curve_location_range)
            )
        elif new_segment_type == SegmentType.STAR:
            start_mode = random.choice(list(StartMode))
            new_segment = create_segment(
                segment_type=SegmentType.STAR,
                start=segment_start,
                start_mode=random.choice([StartMode.CONNECT, StartMode.JUMP]),
                radius=random.uniform(*radius_range),
                arm_length=random.uniform(*arm_length_range),
                num_points=random.randint(*num_points_range),
                asymmetry=random.uniform(*asymmetry_range),
                curved=random.choice([True, False]),
                end_thickness=random.uniform(*thickness_range),
                relative_angle=random.uniform(*direction_range),
                colour=random.choice(colour_options),
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
        fig = gene.render()  # Render each gene on its specific subplot
        delete_segment = analyze_gene(gene)

        while delete_segment:
            gene_pool[idx] = random_gene(idx)
            gene = gene_pool[idx]  # Update the loop variable with the new gene
            ax.clear()  # Clear the previous gene's rendering
            ax.set_title(f"New design {idx + 1}")  # Reset the title
            fig = gene.render()  # Render the new gene
            delete_segment = analyze_gene(gene)

    return gene_pool

def check_overlap(i, segments):
    segment = segments[i].points_array
    overlaps =0
    for j in range (i+1,len(segments)-1):
        overlap_found = False
        segment_j = segments[j].points_array

        if len(segment) >= 1200:
            segment = segment[::2]
        if len(segment_j) >= 1200:
            segment_j = segment_j[::2]

        if j == (i + 1) and (segments[j].start_mode == StartMode.CONNECT_MID or segments[j].start_mode == StartMode.CONNECT ):
            segment_j = segment_j[2:]
        elif j == (i + 1) and (segments[j].start_mode == StartMode.CONNECT or segments[j].start_mode == StartMode.SPLIT):
            segment = segment[:-2]
        for point1 in segment:
            for point2 in segment_j:
                # Calculate Euclidean distance between point1 and point2
                distance = np.linalg.norm(point1 - point2)
                # Check if distance is within the tolerance
                if distance <= 0.05:
                    overlap_found = True
        if overlap_found :
            overlaps += 1  # Overlap found

    return overlaps


def analyze_gene(design):
    segments = design.segments
    score = 0  # Count how many overlaps there are in this gene
    # Compare each pair of segments for overlap
    for i in range(len(segments)):
        score = score - check_overlap(i , segments)
        if i+1<len(segments):
            if segments[i].segment_type == SegmentType.LINE and segments[i+1].segment_type == SegmentType.LINE:
                if segments[i+1].start_mode == StartMode.CONNECT and 90 <= segments[i+1].relative_angle <= 180:
                    score = score + 1
    print("score: ",score)
    if score < -4:
        return True
    else:
        return False

#plt = initialise_gene_pool()
#plt.tight_layout()
#plt.show()

"""
num_points=random.randint(*num_points_range)
new_star_segment = create_segment(
                segment_type=SegmentType.STAR,
                start = (0,0),
                start_mode=StartMode.CONNECT,
                radius=random.uniform(*radius_range),
                arm_length=2,#random.uniform(*arm_length_range),
                num_points=num_points,
                asymmetry= 0,#random.uniform(*asymmetry_range),
                curved=random.choice([True, False]),
                end_thickness=random.uniform(*thickness_range),
                relative_angle = random.uniform(*direction_range),
                end_arm= random.uniform(0,num_points-1)
            )
design.add_segment(new_star_segment)
next_start_thickness = design.get_start_thickness()



for i in range(5):
    new_segment = create_segment(
        segment_type=SegmentType.LINE,
        start = (0,0),
        start_mode=random.choice([StartMode.CONNECT, StartMode.SPLIT, StartMode.CONNECT_MID]),
        length=random.uniform(*length_range),
        relative_angle=random.uniform(*direction_range),
        start_thickness=next_start_thickness,
        end_thickness=random.uniform(*thickness_range),
        color=random.choice(colour_options),
        curviness= 0 if random.random() < 0.5 else random.uniform(*curviness_range),
        curve_direction=random.uniform(*curve_direction_range),
        curve_location=random.uniform(*curve_location_range),
        start_location=random.uniform(*curve_location_range),
        split_point=random.uniform(*curve_location_range)

    )
    print("start mode:", new_segment.start_mode)
    design.add_segment(new_segment)
    next_start_thickness = design.get_start_thickness()
"""

design = EyelinerDesign()
next_start_thickness = random.uniform(1,5)

#Parameter ranges:
start_x_range = (-1,10)
start_y_range = (0,10)
#start_mode_options = list(StartMode)
#Segment_type_options = list(SegmentType)
length_range = (2, 5)
direction_range = (0, 360)
thickness_range = (1, 5)
colour_options = ["red", "green", "blue", "cyan", "magenta"]
curviness_range = (0, 10)
curve_direction_range = (0, 360)
curve_location_range = (0,1)
center_x_range = (-2,2)
center_y_range = (-2,2)
radius_range = (0,2)
arm_length_range = (0,2)
num_points_range = (3,20)
asymmetry_range = (0,3)

new_segment = create_segment(
        segment_type=SegmentType.LINE,
        start = (3,1.5),
        start_mode=StartMode.CONNECT,
        length=random.uniform(*length_range),
        relative_angle=random.uniform(*direction_range),
        start_thickness=next_start_thickness,
        end_thickness=random.uniform(*thickness_range),
        colour="red",
        curviness= 0 if random.random() < 0.5 else random.uniform(*curviness_range),
        curve_direction=random.uniform(*curve_direction_range),
        curve_location=random.uniform(*curve_location_range),
        start_location=random.uniform(*curve_location_range),
        split_point=random.uniform(*curve_location_range)

)
design.add_segment(new_segment)
next_start_thickness = design.get_start_thickness()

new_segment = create_segment(
        segment_type=SegmentType.LINE,
        start = (3,1.5),
        start_mode=StartMode.CONNECT_MID,
        length=random.uniform(*length_range),
        relative_angle=random.uniform(*direction_range),
        start_thickness=next_start_thickness,
        end_thickness=random.uniform(*thickness_range),
        colour="green",
        curviness= 0 if random.random() < 0.5 else random.uniform(*curviness_range),
        curve_direction=random.uniform(*curve_direction_range),
        curve_location=random.uniform(*curve_location_range),
        start_location=random.uniform(*curve_location_range),
        split_point=random.uniform(*curve_location_range)

)
design.add_segment(new_segment)
next_start_thickness = design.get_start_thickness()

new_segment = create_segment(
        segment_type=SegmentType.LINE,
        start = (0,0),
        start_mode=StartMode.SPLIT,
        length=random.uniform(*length_range),
        relative_angle=random.uniform(*direction_range),
        start_thickness=next_start_thickness,
        end_thickness=random.uniform(*thickness_range),
        colour="blue",
        curviness= 0 if random.random() < 0.5 else random.uniform(*curviness_range),
        curve_direction=random.uniform(*curve_direction_range),
        curve_location=random.uniform(*curve_location_range),
        start_location=random.uniform(*curve_location_range),
        split_point=random.uniform(*curve_location_range)

)
design.add_segment(new_segment)
design.render()
plt.show()
