import random
from enum import Enum
min_fitness_score = -20
initial_gene_pool_size = 100
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
curviness_range = (0, 10)
relative_location_range = (0, 1)
radius_range = (0, 1.5)
arm_length_range = (0, 1.5)
num_points_range = (3, 8)
asymmetry_range = (0, 3)

number_of_children_range= (0,3)
max_segments = 20
average_branch_length = 3.5

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

class StartMode(Enum):
    CONNECT = 'CONNECT'
    JUMP = 'JUMP'
    SPLIT = 'SPLIT' #connect but end of prev is in segment
    CONNECT_MID = 'CONNECT_MID'

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