import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import comb

#from dask_expr.diagnostics import analyze

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
    length_range = (0.5, 5)
    direction_range = (0, 360)
    thickness_range = (1, 5)
    colour_options = [
        "red", "green", "blue", "cyan", "magenta",
        "yellow", "black", "white", "orange", "purple",
        "pink", "brown", "gray", "lime", "navy",
        "teal", "maroon", "gold", "silver", "olive"
    ]

    curviness_range = (0, 10)
    curve_direction_range = (0, 360)
    curve_location_range = (0,1)
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
        #new_segment_type = random.choice(list(SegmentType))
        new_segment_type = SegmentType.LINE if random.random() < 0.8 else SegmentType.STAR
        if new_segment_type == SegmentType.LINE:
            new_segment = create_segment(
                segment_type=SegmentType.LINE,
                start=segment_start,
                start_mode=random.choice(list(StartMode)),
                length=random.uniform(*length_range),
                relative_angle=random.uniform(*direction_range),
                start_thickness=next_start_thickness,
                end_thickness=random.uniform(*thickness_range),
                color=random.choice(colour_options),
                curviness = 0 if random.random() < 0.5 else random.uniform(*curviness_range),
                curve_direction=random.uniform(*curve_direction_range),
                curve_location=random.uniform(*curve_location_range),
                start_location=random.uniform(*curve_location_range),
                split_point = random.uniform(*curve_location_range)
            )
        elif new_segment_type == SegmentType.STAR:
            num_points = random.randint(*num_points_range)
            new_segment = create_segment(
                segment_type=SegmentType.STAR,
                start=segment_start,
                start_mode=random.choice([StartMode.CONNECT, StartMode.JUMP]),
                radius=random.uniform(*radius_range),
                arm_length=random.uniform(*arm_length_range),
                num_points=num_points,
                asymmetry=random.uniform(*asymmetry_range),
                curved=random.choice([True, False]),
                end_thickness=random.uniform(*thickness_range),
                relative_angle=random.uniform(*direction_range),
                end_arm=random.randint(0, num_points-1)
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
        analyze_gene(gene)

    plt.tight_layout()
    plt.show()

def check_overlap(i, segments):
    segment = segments[i].points_array
    overlaps =0
    for j in range (len(segments)):
        if j!=i:
            segment_i = segments[j].points_array
            if len(segment) >= 100:
                segment = segment[::2]
            if len(segment_i) >= 100:
                segment_i = segment_i[::2]

            segment = segment[:-1]
            segment_i = segment_i[1:]
            for point1 in segment:
                for point2 in segment_i:
                    # Calculate Euclidean distance between point1 and point2
                    distance = np.linalg.norm(point1 - point2)
                    # Check if distance is within the tolerance
                    if distance <= 0.001:
                        overlaps +=1  # Overlap found
    return overlaps


def analyze_gene(design ):
    segments = design.segments
    score = 0  # Count how many overlaps there are in this gene
    # Compare each pair of segments for overlap
    for i in range(len(segments)):
        score = score - check_overlap(i , segments)
        if i+1<len(segments):
            if segments[i].segment_type == SegmentType.LINE and segments[i+1].segment_type == SegmentType.LINE:
                if segments[i+1].start_mode == StartMode.CONNECT and 0 <= segments[i+1].relative_angle <= 90:
                    score = score + 1
    print("score: ",score)

initialise_gene_pool()


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

plt.figure(figsize=(6, 6))
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlim(-5,10)
ax.set_ylim(-5,10)

"""
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

new_segment = create_segment(
        segment_type=SegmentType.LINE,
        start = (3,1.5),
        start_mode=StartMode.CONNECT,
        length=random.uniform(*length_range),
        relative_angle=random.uniform(*direction_range),
        start_thickness=next_start_thickness,
        end_thickness=random.uniform(*thickness_range),
        color="red",
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
        color="green",
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
        color="blue",
        curviness= 0 if random.random() < 0.5 else random.uniform(*curviness_range),
        curve_direction=random.uniform(*curve_direction_range),
        curve_location=random.uniform(*curve_location_range),
        start_location=random.uniform(*curve_location_range),
        split_point=random.uniform(*curve_location_range)

)
print("start mode:", new_segment.start_mode)
design.add_segment(new_segment)
next_start_thickness = design.get_start_thickness()


design.render(ax)
plt.show()
"""