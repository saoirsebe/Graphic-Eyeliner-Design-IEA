import random
from enum import Enum
import numpy as np

min_fitness_score = -3
min_segment_score = -2
initial_gene_pool_size = 10
node_re_gen_max = 12
#Parameter ranges:
start_x_range = (20, 130)
start_y_range = (50, 150)
#start_mode_options = list(StartMode)
#Segment_type_options = list(SegmentType)
length_range = (10, 50)
direction_range = (0, 360)
thickness_range = (0.5, 3)
colour_options = [
    "red", "green", "blue", "cyan", "magenta", "black", "orange", "purple",
    "pink", "brown", "gray", "lime", "navy",
    "teal", "maroon", "gold", "olive"
]
curviness_range = (0, 1)
relative_location_range = (0, 1)
radius_range = (0, 25)
arm_length_range = (0, 25)
num_points_range = (3, 8)
asymmetry_range = (0, 10)
corner_initialisation_range = (0,1)
random_shape_size_range = (10,40)
max_shape_overlaps = 0

number_of_children_range= (0,3)
max_segments = 20
average_branch_length = 3.5
eye_corner_start = (118.1,93)

line_num_points = 100

def bezier_curve(P0, P1, P2):
    """returns an array of shape (100,2)(100,2), which represents 100 points on the Bézier curve. """
    t = np.linspace(0, 1, 100).reshape(-1, 1)
    return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2

# Function to create a quadratic Bézier curve with three control points
def bezier_curve_t(t, P0, P1, P2):
    """The Bézier curve formulas used in the function calculate the x-coordinate and y-coordinate of a point at a given tt on the curve using the quadratic Bézier formula:"""
    x = (1 - t) ** 2 * P0[0] + 2 * (1 - t) * t * P1[0] + t ** 2 * P2[0]
    y = (1 - t) ** 2 * P0[1] + 2 * (1 - t) * t * P1[1] + t ** 2 * P2[1]

    return [round(x, 3), round(y, 3)]

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
    IRREGULAR_POLYGON = 'IRREGULAR_POLYGON'

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

#upper_x, upper_y = get_quadratic_points(0.0121,-9.1205, -1575.91,  320, 435)
#upper_eyelid_x, upper_eyelid_y = np.array(upper_x) * 3, np.array(upper_y) * 3
#upper_eyelid_coords = np.column_stack((upper_eyelid_x, upper_eyelid_y))
"""
upper_x, upper_y = get_quadratic_points(-0.5, 0, 1, -1, 1)
upper_eyelid_x, upper_eyelid_y = np.array(upper_x) * 3, np.array(upper_y) * 3
upper_eyelid_coords = np.column_stack((upper_eyelid_x, upper_eyelid_y))

lower_x, lower_y = get_quadratic_points(0.5, 0, 0, -1, 1)
lower_eyelid_x, lower_eyelid_y = np.array(lower_x) * 3, np.array(lower_y) * 3
lower_eyelid_coords = np.column_stack((lower_eyelid_x, lower_eyelid_y))
"""

upper_eyelid_coords =  bezier_curve(np.array([28,82]), np.array([70,123]), np.array([118,93]))
upper_eyelid_x = [coord[0] for coord in upper_eyelid_coords]
upper_eyelid_y = [coord[1] for coord in upper_eyelid_coords]

lower_eyelid_coords =  bezier_curve(np.array([28,82]), np.array([100,60]), np.array([118,93]))
lower_eyelid_x = [coord[0] for coord in lower_eyelid_coords]
lower_eyelid_y = [coord[1] for coord in lower_eyelid_coords]
"""

lower_x, lower_y = get_quadratic_points(-0.0121, 9.1205, -1576.91,  320, 435)
lower_eyelid_x, lower_eyelid_y = np.array(lower_x) * 3, np.array(lower_y) * 3
lower_eyelid_coords = np.column_stack((lower_eyelid_x, lower_eyelid_y))

upper_x_left, upper_y_left = get_quadratic_points(0.000185,-0.240, 99.69, 5, 221)
upper_eyelid_x_left, upper_eyelid_y_left = np.array(upper_x) * 3, np.array(upper_y) * 3
upper_eyelid_coords_left = np.column_stack((upper_eyelid_x, upper_eyelid_y))

lower_x_left, lower_y_left = get_quadratic_points(0.00133, -0.351, 106.58, 5, 221)
lower_eyelid_x_left, lower_eyelid_y_left = np.array(lower_x) * 3, np.array(lower_y) * 3
lower_eyelid_coords_left = np.column_stack((lower_eyelid_x, lower_eyelid_y))
"""
def draw_eye_shape(ax_n):
    ax_n.plot(lower_eyelid_x, lower_eyelid_y, label=f"$y = 0.5x^2$", color="b")
    #ax_n.plot(upper_x, upper_y, label=f"$y = -0.5x^2 + 1$", color="red")
    ax_n.plot(upper_eyelid_x, upper_eyelid_y, label=f"$y = -0.5x^2 + 1$", color="green")



