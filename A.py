import random
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageTk

min_fitness_score = -3
min_segment_score = -2
initial_gene_pool_size = 200
node_re_gen_max = 12
#Parameter ranges:
start_x_range = (20, 140)
start_y_range = (50, 130)
#start_mode_options = list(StartMode)
#Segment_type_options = list(SegmentType)
length_range = (15, 50)
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


upper_eyelid_coords =  bezier_curve(np.array([28,82]), np.array([70,123]), np.array([118,93]))
upper_eyelid_x = [coord[0] for coord in upper_eyelid_coords]
upper_eyelid_y = [coord[1] for coord in upper_eyelid_coords]

lower_eyelid_coords =  bezier_curve(np.array([28,82]), np.array([100,60]), np.array([118,93]))
lower_eyelid_x = [coord[0] for coord in lower_eyelid_coords]
lower_eyelid_y = [coord[1] for coord in lower_eyelid_coords]

def draw_eye_shape(ax_n):
    ax_n.plot(lower_eyelid_x, lower_eyelid_y, label=f"$y = 0.5x^2$", color="b")
    #ax_n.plot(upper_x, upper_y, label=f"$y = -0.5x^2 + 1$", color="red")
    ax_n.plot(upper_eyelid_x, upper_eyelid_y, label=f"$y = -0.5x^2 + 1$", color="green")


face_end_x_values = np.linspace(138, 180, line_num_points*2)
face_end_y_values = np.linspace(0, 175, line_num_points*2)
face_end = np.column_stack((face_end_x_values, face_end_y_values))
face_end = np.round(face_end, 3)

def generate_eyeliner_curve_lines(ax_n=None):
    #plt.figure(figsize=(6, 6))
    if ax_n:
        draw_eye_shape(ax_n)

    x_values = np.linspace(eye_corner_start[0], 160, 100)
    y_values = np.linspace(eye_corner_start[1], 125, 100)
    top_eye_curve = np.column_stack((x_values, y_values))
    top_eye_curve -= 0.5

    #top_eye_curve = bezier_curve(P0,P1,P2)
    if ax_n:
        flipped_img = np.flipud(Image.open("female-face-drawing-template-one-eye.jpg"))
        ax_n.imshow(flipped_img)
        ax_n.invert_yaxis()
        ax_n.plot(top_eye_curve[:,0], top_eye_curve[:,1], label="Upper Bezier")

    P0 = np.array(eye_corner_start)
    P2 = np.array([160, 105])
    P1 = (P0 + P2) // 2
    P1[1] -= 5
    P1[0] += 5

    bottom_eye_curve = bezier_curve(P0,P1,P2)
    #bottom_eye_curve = np.column_stack((x, y))
    if ax_n:
        ax_n.plot(bottom_eye_curve[:,0], bottom_eye_curve[:,1], label="Lower Bezier")
    #ax = plt.gca()
    #ax.set_aspect('equal')
    #plt.show()
    return top_eye_curve, bottom_eye_curve

#fig, ax = plt.subplots()
eyeliner_curve1, eyeliner_curve2 = generate_eyeliner_curve_lines()
#fig.show()

