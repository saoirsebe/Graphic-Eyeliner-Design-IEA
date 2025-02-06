import random
from enum import Enum

import numpy as np

min_fitness_score = -3
min_segment_score = -2
initial_gene_pool_size = 200
node_re_gen_max = 12
#Parameter ranges:
start_x_range = (-1, 7)
start_y_range = (0, 6)
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
curviness_range = (0.1, 10)
relative_location_range = (0, 1)
radius_range = (0, 1.5)
arm_length_range = (0, 1.5)
num_points_range = (3, 8)
asymmetry_range = (0, 3)
edge_initialisation_range = (0,1)
random_shape_size_range = (1,5)
max_shape_overlaps = 0

number_of_children_range= (0,3)
max_segments = 20
average_branch_length = 3.5
eye_corner_start = (3,1.5)

class StarType(Enum):
    STRAIGHT = 'STRAIGHT'
    CURVED = 'CURVED'
    FLOWER = 'FLOWER'

class SegmentType(Enum):
    LINE = 'LINE'
    #FORK = 'FORK'
    #TAPER = 'TAPER'
    STAR = 'STAR'
    #WING = 'WING'
    RANDOM_SHAPE = 'RANDOM_SHAPE'

class StartMode(Enum):
    CONNECT = 'CONNECT'
    JUMP = 'JUMP'
    SPLIT = 'SPLIT' #connect but end of prev is in segment
    CONNECT_MID = 'CONNECT_MID'



def get_quadratic_points(a, b, c, x_start, x_end, num_points=100):
    x = np.linspace(x_start, x_end, num_points)  # Generate x values from x_start to x_end
    y = a * x ** 2 + b * x + c  # Calculate corresponding y values
    return x, y

def un_normalised_vector_direction(start,end):
    direction = end - start

    return direction
def normalised_vector_direction(start,end):
    direction = end - start
    norm = np.linalg.norm(direction) #length of direction vector
    if norm != 0:
        direction /= norm  # Normalize the direction vector
    return direction

def random_normal_within_range(mean, stddev, value_range):
    while True:
        # Generate a number using normal distribution
        value = random.gauss(mean, stddev)
        # Keep it within the specified range
        if value_range[0] <= value <= value_range[1]:
            return value

def random_from_two_distributions(mean1, stddev1, mean2, stddev2, value_range, prob1=0.5):
    while True:
        # Choose which distribution to use
        if random.random() < prob1:
            value = random.gauss(mean1, stddev1)
        else:  # Otherwise, use the second distribution
            value = random.gauss(mean2, stddev2)

        if value_range[0] <= value <= value_range[1]:
            return value

def draw_eye_shape(ax_n):
    # Get points with the same x-range but scale y-values for vertical stretch
    x_vals, y_vals = get_quadratic_points(0.5, 0, 0, -1, 1)
    x_vals = [x * 3 for x in x_vals]
    y_vals = [y * 3 for y in y_vals]  # Scale y-values
    ax_n.plot(x_vals, y_vals, label=f"$y = 0.5x^2$", color="b")

    x_vals, y_vals = get_quadratic_points(-0.5, 0, 1, -1, 1)
    x_vals = [x * 3 for x in x_vals]
    y_vals = [y * 3 for y in y_vals]  # Scale y-values
    ax_n.plot(x_vals, y_vals, label=f"$y = -0.5x^2 + 1$", color="b")

upper_x, upper_y = get_quadratic_points(-0.5, 0, 1, -1, 1)
upper_eyelid_x, upper_eyelid_y = np.array(upper_x) * 3, np.array(upper_y) * 3
upper_eyelid_coords = np.column_stack((upper_eyelid_x, upper_eyelid_y))

lower_x, lower_y = get_quadratic_points(0.5, 0, 0, -1, 1)
lower_eyelid_x, lower_eyelid_y = np.array(lower_x) * 3, np.array(lower_y) * 3
lower_eyelid_coords = np.column_stack((lower_eyelid_x, lower_eyelid_y))