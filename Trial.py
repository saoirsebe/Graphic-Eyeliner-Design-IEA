import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import comb
from EAOverallGenome import *
import random



def random_gene():
    design = EyelinerDesign()
    n_objects = random.randint(1,10)
    next_start_thickness = random.uniform(1,5)

    #Parameter ranges:
    start_x_range = (-2,2)
    start_y_range = (-2,2)
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
    radius_range = (0,3)
    arm_length_range = (0,3)
    num_points_range = (3,20)
    asymmetry_range = (0,3)
    #curved_range = ["True","False"]

    for i in range(n_objects):
        # Adding a new segment to the design
        new_segment_type = random.choice(list(SegmentType))
        start_type = random.choice(list(StartMode))
        print(start_type)
        if new_segment_type == SegmentType.LINE:
            new_segment = create_segment(
                segment_type=new_segment_type,
                start = (random.uniform(*start_x_range), random.uniform(*start_y_range)),
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
                start=(random.uniform(*start_x_range), random.uniform(*start_y_range)),
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

    design.render()
    return design

makeup_design = random_gene()
