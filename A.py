from enum import Enum
min_fitness_score = -30
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
    BRANCH_POINT = 'BRANCH_POINT'
    END_POINT = 'END_POINT'

class StartMode(Enum):
    CONNECT = 'CONNECT'
    JUMP = 'JUMP'
    SPLIT = 'SPLIT' #connect but end of prev is in segment
    CONNECT_MID = 'CONNECT_MID'