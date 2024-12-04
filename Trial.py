import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import comb
#from dask_expr.diagnostics import analyze
from EAOverallGenome import *
import random
from EyelinerWingGeneration import get_quadratic_points
from A import *


def random_gene(gene_n):
    design = EyelinerDesign()
    n_objects = random.randint(2, 10)
    for i in range(n_objects):
        if i == 0:  #(gene_n ==0 and i==0) or (gene_n ==1 and i==0):
            segment_start = (3, 1.5)
        else:
            segment_start = (random.uniform(*start_x_range), random.uniform(*start_y_range))

        # Adding a new segment to the design
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
                curviness=0 if random.random() < 0.5 else random.uniform(*curviness_range),
                curve_direction=random.uniform(*direction_range),
                curve_location=random.uniform(*relative_location_range),
                start_location=random.uniform(*relative_location_range),
                split_point=random.uniform(*relative_location_range)
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
                star_type=random.choice([StarType.STRAIGHT,StarType.CURVED,StarType.FLOWER]),
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
        plt.close(fig)
        delete_segment = analyze_gene(gene)
        while delete_segment:
            gene_pool[idx] = random_gene(idx)
            gene = gene_pool[idx]  # Update the loop variable with the new gene
            ax.clear()  # Clear the previous gene's rendering
            ax.set_title(f"New design {idx + 1}")  # Reset the title
            fig = gene.render()  # Render the new gene
            delete_segment = analyze_gene(gene)
            plt.close(fig)
    return gene_pool


def check_overlap(i, segments):
    segment = segments[i].points_array
    overlaps = 0
    for j in range(i + 1, len(segments) - 1):
        overlap_found = False
        segment_overlaps = 0
        segment_j = segments[j].points_array

        if j == (i + 1) and (
                segments[j].start_mode == StartMode.CONNECT_MID or segments[j].start_mode == StartMode.CONNECT):
            first_1 = int(len(segment_j) * 0.05)
            segment_j = segment_j[first_1:]
        elif j == (i + 1) and (
                segments[j].start_mode == StartMode.CONNECT or segments[j].start_mode == StartMode.SPLIT):
            first_1 = int(len(segment) * 0.05)
            segment = segment[:-first_1]
        for point1 in segment:
            for point2 in segment_j:
                # Calculate Euclidean distance between point1 and point2
                distance = np.linalg.norm(point1 - point2)
                # Check if distance is within the tolerance
                if distance <= 0.075:
                    overlap_found = True
                    segment_overlaps += 1
        if overlap_found:
            overlaps += int((segment_overlaps / (len(segment) + len(segment_j))) * 100)

    return overlaps


def is_in_eye(segment):
    overlap = 0
    # Get eye shape boundary points
    upper_x, upper_y = get_quadratic_points(-0.5, 0, 1, -1, 1)
    lower_x, lower_y = get_quadratic_points(0.5, 0, 0, -1, 1)
    upper_x, upper_y = np.array(upper_x) * 3, np.array(upper_y) * 3
    lower_x, lower_y = np.array(lower_x) * 3, np.array(lower_y) * 3

    segment = segment.points_array
    upper_y_interp = np.interp(segment[:, 0], upper_x, upper_y)
    lower_y_interp = np.interp(segment[:, 0], lower_x, lower_y)
    inside = (lower_y_interp <= segment[:, 1]) & (segment[:, 1] <= upper_y_interp)
    overlap = np.sum(inside)
    score = int((overlap / len(segment)) * 100)
    if score > 0:
        print("In eyee:", score)
        return max(score, 1)
    else:
        return 0


def wing_angle(i, segments):
    if i + 1 < len(segments):
        if segments[i].segment_type == SegmentType.LINE and segments[i + 1].segment_type == SegmentType.LINE:
            if segments[i + 1].start_mode == StartMode.CONNECT and (80 < segments[i + 1].relative_angle < 170 or 190 < segments[i + 1].relative_angle < 260):
                return 5
    return 0


def analyze_gene(design):
    segments = design.segments
    score = 0  # Count how many overlaps there are in this gene
    # Compare each pair of segments for overlap
    for i in range(len(segments)):
        score = score - check_overlap(i, segments)
        score = score + wing_angle(i, segments)
        score = score - is_in_eye(segments[i])
    print("score: ", score)
    if score <= -20:
        return True
    else:
        return False


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
"""
design = EyelinerDesign()
new_segment = create_segment(
        segment_type=SegmentType.LINE,
        start = (3,1.5),
        start_mode=StartMode.CONNECT,
        length=3,
        relative_angle=40,
        start_thickness=2.5,
        end_thickness=1,
        colour="red",
        curviness= 0.2 ,
        curve_direction=0.2,
        curve_location=0.5,
        start_location=0.6,
        split_point=0.2

)
design.add_segment(new_segment)
new_segment = create_segment(
        segment_type=SegmentType.LINE,
        start = (3,1.5),
        start_mode=StartMode.CONNECT_MID,
        length=1.5,
        relative_angle=110,
        start_thickness=6,
        end_thickness=4,
        colour="orange",
        curviness= 0 ,
        curve_direction=0.2,
        curve_location=0.5,
        start_location=0.6,
        split_point=0.2

)
design.add_segment(new_segment)
new_segment = create_segment(
        segment_type=SegmentType.LINE,
        start = (3,1.5),
        start_mode=StartMode.SPLIT,
        length=7,
        relative_angle=20,
        start_thickness=2.5,
        end_thickness=1,
        colour="pink",
        curviness= 0 ,
        curve_direction=0,
        curve_location=0.5,
        start_location=0.6,
        split_point=0.2

)
design.add_segment(new_segment)
fig = design.render()
fig.show()


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
"""
"""
line_seg = LineSegment(SegmentType.LINE,(0,0),StartMode.JUMP,5,230,2,4,"purple",0,0,0.8,0,0)
#design.render()
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_aspect('equal')
line_seg.render(ax,[],0,"white",3)
plt.show()

"""
star_seg = StarSegment(SegmentType.STAR,(0,0),"pink",StarType.STRAIGHT,1,2,6,0,StartMode.JUMP,3,30)
# curviness of star FIX!!!!!!!!!
#Asymetry broken FIX!!!
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_aspect('equal')
star_seg.render(ax,[],0,"white",3)
plt.show()

