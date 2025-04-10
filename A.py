import random
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
from numba import njit

diff_threshold = 12.5
min_negative_score = -3
min_segment_score = -2
initial_gene_pool_size = 50
node_re_gen_max = 12
#Parameter ranges:
start_x_range = (20, 130)
start_y_range = (55, 120)
#start_mode_options = list(StartMode)
#Segment_type_options = list(SegmentType)
length_range = (15, 60)
direction_range = (0, 360)
thickness_range = (0.5, 3)

colour_options = [
    "red", "orange", "yellow", "green", "blue", "indigo", "violet",
    "lightblue", "pink", "purple", "brown", "grey", "black", "cyan", "magenta",
    "maroon", "navy", "olive", "teal", "darkred", "darkgreen", "darkblue",
    "darkorange", "darkviolet", "darkgrey"
]

dark_colours = [
    "brown", "grey", "black", "maroon", "navy", "olive", "teal",
    "darkred", "darkgreen", "darkblue", "darkorange", "darkviolet", "darkgrey"
]


similar_colours = {
    "red": ["orange", "magenta", "pink", "darkred", "maroon", "brown", "black"],
    "orange": ["red", "yellow", "darkorange", "brown", "black"],
    "yellow": ["orange", "green", "olive", "black"],
    "green": ["yellow", "blue", "cyan", "olive", "darkgreen", "teal", "black"],
    "blue": ["green", "indigo", "lightblue", "navy", "darkblue", "cyan", "black"],
    "indigo": ["blue", "violet", "purple", "navy", "darkblue", "black"],
    "violet": ["indigo", "purple", "magenta", "darkviolet", "black"],
    "lightblue": ["blue", "cyan", "navy", "black"],
    "pink": ["red", "magenta", "purple", "darkred", "black"],
    "purple": ["violet", "pink", "indigo", "darkviolet", "black"],
    "brown": ["red", "orange", "maroon", "darkred", "darkorange", "black"],
    "grey": ["darkgrey", "black"],
    "black": ["red", "orange", "yellow", "green", "blue", "indigo", "violet",
        "lightblue", "pink", "purple", "brown", "grey", "black", "cyan", "magenta",
        "maroon", "navy", "olive", "teal", "darkred", "darkgreen", "darkblue",
        "darkorange", "darkviolet", "darkgrey"],
    "cyan": ["blue", "lightblue", "green", "teal", "darkblue", "black"],
    "magenta": ["red", "pink", "violet", "purple", "darkred", "black"],
    "maroon": ["red", "darkred", "brown", "black"],
    "navy": ["blue", "indigo", "darkblue", "black"],
    "olive": ["green", "yellow", "darkgreen", "brown", "black"],
    "teal": ["blue", "green", "cyan", "darkblue", "darkgreen", "black"],
    "darkred": ["red", "maroon", "brown", "magenta", "black"],
    "darkgreen": ["green", "olive", "teal", "darkblue", "black"],
    "darkblue": ["blue", "navy", "indigo", "teal", "black"],
    "darkorange": ["orange", "red", "brown", "black"],
    "darkviolet": ["violet", "purple", "indigo", "black"],
    "darkgrey": ["grey", "black"],
}

curviness_range = (0, 1)
relative_location_range = (0, 1)
radius_range = (0, 25)
arm_length_range = (5, 25)
num_points_range = (3, 8)
asymmetry_range = (0, 10)
corner_initialisation_range = (0,1)
random_shape_size_range = (10,50)
max_shape_overlaps = 0

number_of_children_range= (0,3)
max_segments = 20
average_branch_length = 5


line_num_points = 100

def bezier_curve(P0, P1, P2):
    """returns an array of shape (100,2)(100,2), which represents 100 points on the Bézier curve. """
    t = np.linspace(0, 1, 100).reshape(-1, 1)
    return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2


@njit
def bezier_curve_t(t_array, P0, P1, P2):
    n = t_array.shape[0]
    d = P0.shape[0]
    result = np.empty((n, d))
    for i in range(n):
        t = t_array[i]
        u = 1 - t
        for j in range(d):
            result[i, j] = u*u * P0[j] + 2*u*t * P1[j] + t*t * P2[j]
    return np.round(result, 3)

class StarType(Enum):
    STRAIGHT = 'STRAIGHT'
    CURVED = 'CURVED'
    FLOWER = 'FLOWER'

class SegmentType(Enum):
    LINE = 'LINE'
    STAR = 'STAR'
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

def set_prev_array(parent):
    if parent.segment_type == SegmentType.STAR:
        return parent.arm_points_array # if segment is a star then pass in the arm points that the next segment should start at:
    elif parent.segment_type == SegmentType.LINE:
        return parent.points_array
    else:
        if not parent.is_eyeliner_wing:
            return parent.to_scale_corners
        else:
            if len(parent.corners) ==3:
                return np.array([parent.corners[1]])#parent.to_scale_corners[1]])
            elif len(parent.corners) ==4:
                return np.array([parent.corners[2]])#parent.to_scale_corners[2]])
            else:
                raise ValueError("Wrong number of corners for eyeliner wing")

upper_eyelid_coords =  bezier_curve(np.array([29,80]), np.array([70,123]), np.array([118,93]))
upper_eyelid_x = [coord[0] for coord in upper_eyelid_coords]
upper_eyelid_y = [coord[1] for coord in upper_eyelid_coords]
upper_len = len(upper_eyelid_coords)
upper_eyelid_coords_40 = int(0.4 * upper_len)
upper_eyelid_coords_10 = int(0.05 * upper_len)
#for p2 mutation:
abs_min = upper_len - upper_eyelid_coords_40
abs_max = upper_len - upper_eyelid_coords_10
allowed_indices = np.arange(abs_min, abs_max + 1)

lower_eyelid_coords =  bezier_curve(np.array([29,80]), np.array([98,62]), np.array([118,93]))
lower_eyelid_x = [coord[0] for coord in lower_eyelid_coords]
lower_eyelid_y = [coord[1] for coord in lower_eyelid_coords]
lower_len = len(lower_eyelid_coords)
lower_eyelid_coords_section = int(0.1 * lower_len)

eye_corner_start = (118.25,93)

def draw_eye_shape(ax_n):
    ax_n.plot(lower_eyelid_x, lower_eyelid_y, label=f"$y = 0.5x^2$", color="black")
    #ax_n.plot(upper_x, upper_y, label=f"$y = -0.5x^2 + 1$", color="red")
    ax_n.plot(upper_eyelid_x, upper_eyelid_y, label=f"$y = -0.5x^2 + 1$", color="black")


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
        ax_n.plot(top_eye_curve[:,0], top_eye_curve[:,1], color='black',label="Upper Bezier")

    P0 = np.array(eye_corner_start)
    P2 = np.array([160, 105])
    P1 = (P0 + P2) // 2
    P1[1] -= 5
    P1[0] += 5

    bottom_eye_curve = bezier_curve(P0,P1,P2)
    #bottom_eye_curve = np.column_stack((x, y))
    if ax_n:
        ax_n.plot(bottom_eye_curve[:,0], bottom_eye_curve[:,1], color='black', label="Lower Bezier")
    #ax = plt.gca()
    #ax.set_aspect('equal')
    #plt.show()
    return top_eye_curve, bottom_eye_curve

def generate_middle_curve_lines(ax_n=None):
    #plt.figure(figsize=(6, 6))
    if ax_n:
        draw_eye_shape(ax_n)

    p0 = np.array([100,115])
    p2 = np.array([140, 125])
    p1 = (p0 + p2) // 2
    p1[1] -= 5
    p1[0] += 5

    curve_upper = bezier_curve(p0, p1, p2)

    if ax_n:
        ax_n.plot(curve_upper[:,0], curve_upper[:,1],color='black', label="Lower Bezier")

    P0 = np.array([100, 60])
    P2 = np.array([140, 75])
    P1 = (P0 + P2) // 2
    P1[1] -= 5
    #P1[0] += 5

    curve_lower = bezier_curve(P0,P1,P2)

    if ax_n:
        ax_n.plot(curve_lower[:,0], curve_lower[:,1],color='black', label="Lower Bezier")
    #ax = plt.gca()
    #ax.set_aspect('equal')
    #plt.show()
    return curve_upper, curve_lower

#fig, ax = plt.subplots()
eyeliner_curve1, eyeliner_curve2 = generate_eyeliner_curve_lines()

middle_curve_upper , middle_curve_lower = generate_middle_curve_lines()

#fig.show()