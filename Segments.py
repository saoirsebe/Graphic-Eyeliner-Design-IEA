from AnalyseDesign import shape_overlaps
import StarGeneration
from ParentSegment import Segment, point_in_array
from RandomShapeSegment import RandomShapeSegment, RandomShapeLineSegment, are_points_collinear
from StarGeneration import *
from A import *
import random

class LineSegment(Segment):
    """Line segment with additional properties specific to a line."""
    def __init__(self, segment_type, start, start_mode, length, relative_angle, start_thickness, end_thickness, colour, curviness, curve_left, curve_location, start_location, split_point):
        super().__init__(segment_type, start, start_mode, end_thickness, relative_angle, colour)
        self.segment_type = SegmentType.LINE
        self.end = None  # Calculated in render
        self.length = length
        self.start_thickness = start_thickness
        self.curviness = curviness
        self.curve_left = curve_left
        if curviness>0:  #If curved line then curve direction else no curve direction
            self.curve_location = curve_location
        else:
            self.curve_location = 0
        if start_mode == StartMode.CONNECT_MID:
            self.start_location = start_location
            self.split_point = 0
        elif start_mode == StartMode.SPLIT:
            self.split_point = split_point  #if mode is split then set split_point else splits from 0 (i.e. no split)
            self.start_location = 1
        else:
            self.start_location = 1
            self.split_point = 0
        self.thickness_array = []

    def calculate_end(self, prev_angle):
        """Calculate the end point based on start, length, and direction."""
        if self.start_mode == StartMode.JUMP:
            self.absolute_angle = self.relative_angle
        else:
            self.absolute_angle = (prev_angle + self.relative_angle) % 360
        radians = math.radians(self.absolute_angle) # Convert to radians

        end_x = self.start[0] + self.length * math.cos(radians)
        end_y = self.start[1] + self.length * math.sin(radians)
        self.end = (end_x, end_y)


    def curve_between_lines(self, p0, p1, p2, p3, colour):
        t_values = np.linspace(0, 1, 20)
        left_curve = np.array([bezier_curve_t(t, p0, p1, p2) for t in t_values])
        # np.array((prev_array[start_array_point_index+2]) - np.array(self.points_array[2]))/2
        right_curve = np.array([bezier_curve_t(t, p2, p1, p3) for t in t_values])
        left_x, left_y = left_curve[:, 0], left_curve[:, 1]
        right_x, right_y = right_curve[:, 0], right_curve[:, 1]

        # Combine the points to form a closed boundary
        boundary_x = np.concatenate([left_x, right_x[::1]])  # Combine left and reversed right x
        boundary_y = np.concatenate([left_y, right_y[::1]])  # Combine left and reversed right y
        plt.fill(boundary_x, boundary_y, color=colour, alpha=0.5)
        return np.concatenate((left_curve, right_curve, self.points_array), axis=0)


    def render(self, prev_array, prev_angle, prev_colour, prev_thickness_array, ax_n=None):
        new_array = []
        num_steps = 50  # Number of points to create a smooth thickness transition/ curve
        if self.start_mode == StartMode.CONNECT and len(prev_array)>15 or self.start_mode == StartMode.SPLIT and len(prev_array)>15:
            self.start = (prev_array[-1][0], prev_array[-1][1])
            if prev_thickness_array.size == 1:
                print("prev_thickness_array:",prev_thickness_array)
                print("prev_array:",prev_array)
            self.start_thickness = prev_thickness_array[len(prev_array) - 1]
        elif self.start_mode == StartMode.CONNECT and len(prev_array)<=15 or self.start_mode == StartMode.SPLIT and len(prev_array)<=15:
            end_index = point_in_array(prev_array, 0.5)
            self.start = (prev_array[end_index][0], prev_array[end_index][1])
        elif len(prev_array)<=0:
            raise ValueError("len(prev_array)<=0....")
        elif self.start_mode == StartMode.CONNECT_MID and len(prev_array)>0:
            start_array_point_index = point_in_array(prev_array, self.start_location)
            start_array_point = prev_array[start_array_point_index]
            self.start = (start_array_point[0], start_array_point[1])

        self.calculate_end(prev_angle)  # Calculate the endpoint

        if self.curviness>0:
            t_values = np.linspace(0, 1, num_steps)
            p0 = np.array(self.start)
            p2 = np.array(self.end)
            p1 = p0 + (self.curve_location * un_normalised_vector_direction(p0,p2)) #moves curve_location away from P0 towards P2 relative to length of curve segment
            if self.curve_left:
                curve_direction = 90 #curve direction in degrees
            else:
                curve_direction = -90
            relative_curve_direction = self.absolute_angle + curve_direction
            curve_dir_radians = np.radians(relative_curve_direction)
            # Calculate x and y offsets
            dx = self.curviness * np.cos(curve_dir_radians)
            dy = self.curviness * np.sin(curve_dir_radians)
            p1 = p1 + np.array([dx, dy])
            self.points_array = np.array([bezier_curve_t(t, p0, p1, p2) for t in t_values])
            x_values, y_values = self.points_array[:, 0], self.points_array[:, 1]
        else:
            x_values = np.linspace(self.start[0], self.end[0], num_steps)
            y_values = np.linspace(self.start[1], self.end[1], num_steps)
            self.points_array = np.column_stack((x_values, y_values))
            self.points_array = np.round(self.points_array, 3)

        """Set self.points_array and move line if split type"""
        if self.start_mode == StartMode.SPLIT:
            split_point_point_index = point_in_array(self.points_array, self.split_point)
            split_point_point = self.points_array[split_point_point_index]
            transformation_vector = un_normalised_vector_direction(split_point_point,self.start)

            self.points_array = np.array([(point[0] + transformation_vector[0], point[1] + transformation_vector[1]) for point in self.points_array])
            x_values, y_values = self.points_array[:, 0], self.points_array[:, 1]

            """Add curve from prev line to split point"""
            if len(prev_array) > 20:
                p2 = np.array(prev_array[-10])
                if split_point_point_index <= 5:
                    beep = 0
                else:
                    beep = split_point_point_index - 5
                p0 = np.array(self.points_array[beep])
                if len(self.points_array) <= (split_point_point_index + 5):
                    beep = len(prev_array) - 1
                else:
                    beep = split_point_point_index + 5
                p3 = np.array(self.points_array[beep])
                p1 = np.array(self.start)
                new_array = self.curve_between_lines(p0, p1, p2, p3, prev_colour)
                x_values, y_values = new_array[:, 0], new_array[:, 1]

        self.thickness_array = np.linspace(self.start_thickness, self.end_thickness, num_steps) #Render a line segment with thickness tapering from start to end
        """Add curve from bottom of connect mid line"""
        if self.start_mode == StartMode.CONNECT_MID and len(prev_array)>20:
            p2 = np.array(self.points_array[10])
            if start_array_point_index <= 5:
                beep = 0
            else:
                beep = start_array_point_index - 5
            p0 = np.array(prev_array[beep])
            if len(prev_array) <= (start_array_point_index + 5):
                beep = len(prev_array) - 1
            else:
                beep = start_array_point_index + 5
            p3 = np.array(prev_array[beep])
            p1 = np.array(self.start)
            new_array = self.curve_between_lines(p0, p1, p2, p3, self.colour)
            x_values, y_values = new_array[:, 0], new_array[:, 1]

        # Plot each small segment with the varying thickness
        if len(new_array) > 0:
            # Plot the first 40 points as one segment
            if self.start_mode == StartMode.CONNECT_MID:
                blend_thicknesses = np.linspace(prev_thickness_array[start_array_point_index], self.thickness_array[10], 20)
            for i in range(min(39, len(x_values) - 1)):
                this_colour = self.colour
                if self.start_mode == StartMode.CONNECT_MID:
                    if i<20:
                        thickness = blend_thicknesses[i]
                    else:
                        thickness = blend_thicknesses[-(i-20)]
                elif self.start_mode == StartMode.SPLIT:
                    thickness = self.thickness_array [0]
                    this_colour = prev_colour
                else:
                    thickness = self.thickness_array [i]

                if ax_n:
                    ax_n.plot(
                        [x_values[i], x_values[i + 1]], [y_values[i], y_values[i + 1]],
                        color=this_colour,
                        linewidth=thickness,
                        solid_capstyle='butt',
                    )

            # Plot remaining points
            for i in range(40, len(x_values) - 1):
                if self.start_mode == StartMode.CONNECT_MID or self.start_mode == StartMode.SPLIT:
                    thickness = self.thickness_array[i - 41]
                else:
                    thickness = self.thickness_array[i]

                if ax_n:
                    ax_n.plot(
                        [x_values[i], x_values[i + 1]], [y_values[i], y_values[i + 1]],
                        color=self.colour,
                        linewidth=thickness,
                        solid_capstyle='butt',
                    )
        else:
            # Plot normally if no new_array exists (Start type is jump or connect so no blend between lines needed)
            for i in range(len(x_values) - 1):
                thickness = self.thickness_array[i]
                if ax_n:
                    ax_n.plot(
                        [x_values[i], x_values[i + 1]], [y_values[i], y_values[i + 1]],
                        color=self.colour,
                        linewidth=thickness,
                        solid_capstyle='butt',
                    )


    def mutate(self, mutation_rate=0.05):
        """Randomly mutate properties of the line segment within a mutation rate."""
        # mutation rate chance of changing:
        self.mutate_choice(self.start_mode, [StartMode.CONNECT, StartMode.JUMP, StartMode.SPLIT, StartMode.CONNECT_MID], mutation_rate)

        if self.start_mode == StartMode.JUMP:
            self.start = (self.mutate_val(self.start[0],start_x_range,mutation_rate),self.mutate_val(self.start[1],start_y_range, mutation_rate))

        if self.start_mode == StartMode.CONNECT_MID:
            self.start_location = self.mutate_val(self.start_location,relative_location_range,mutation_rate)
            self.split_point = 0
        elif self.start_mode == StartMode.SPLIT:
            self.split_point = self.mutate_val(self.split_point, relative_location_range, mutation_rate)
            self.start_location = 1
        else:
            self.start_location = 1
            self.split_point = 0

        self.end_thickness = self.mutate_val(self.end_thickness,thickness_range,mutation_rate)
        self.relative_angle =  self.mutate_val(self.relative_angle,direction_range,mutation_rate)
        self.colour = self.mutate_choice(self.colour, colour_options, mutation_rate)
        self.length = self.mutate_val(self.length,length_range,mutation_rate)
        self.start_thickness = self.mutate_val(self.start_thickness, thickness_range, mutation_rate)

        if self.curviness ==0:
            self.curviness = 0 if np.random.random() > mutation_rate else self.mutate_val(self.curviness,curviness_range,mutation_rate)
        else:
            self.curviness = self.mutate_val(self.curviness,curviness_range,mutation_rate)

        if self.curviness > 0:
            self.curve_direction = self.mutate_val(self.curve_direction,direction_range,mutation_rate)
            self.curve_location = self.mutate_val(self.curve_location,relative_location_range,mutation_rate)
        else:
            self.curve_direction = 0
            self.curve_location = 0


class StarSegment(Segment):
    """Line segment with additional properties specific to a line."""
    def __init__(self, segment_type, start, colour, star_type, radius, arm_length, num_points, asymmetry, start_mode, end_thickness, relative_angle):
        super().__init__(segment_type, start, start_mode, end_thickness,relative_angle, colour)
        self.segment_type = SegmentType.STAR
        self.center = self.start
        self.radius = radius
        self.arm_length = arm_length
        self.num_points = num_points
        self.asymmetry = asymmetry
        self.star_type  = star_type
        #start_angle = 2 * np.pi * num_points / num_points #angle for last arm
        #bin, end_p = StarGeneration.create_star_arm(center, radius,arm_length, num_points,start_angle,asymmetry,num_points,curved) #armN = last arm so num_points
        self.arm_points_array = []

    def render(self, prev_array, prev_angle, prev_colour, prev_end_thickness, ax_n=None):
        if self.start_mode == StartMode.CONNECT and len(prev_array)>15:
            self.start = (prev_array[-1][0], prev_array[-1][1])
            self.center = self.start
        elif self.start_mode == StartMode.CONNECT and len(prev_array)<=15:
            end_index = point_in_array(prev_array, 0.5)
            self.start = (prev_array[end_index][0], prev_array[end_index][1])
            self.center = self.start
        #elif self.start_mode == StartMode.CONNECT_MID and prev_array:

        if self.start_mode == StartMode.JUMP:
            self.absolute_angle = self.relative_angle
        else:
            self.absolute_angle = (prev_angle + self.relative_angle)%360

        star_points, star_arm_points = StarGeneration.create_star(self.num_points, self.center, self.radius, self.arm_length, self.asymmetry, self.star_type, self.absolute_angle)
        start_coord = star_arm_points[-1]
        transformation_vector = (self.center[0] - start_coord[0], self.center[1] - start_coord[1])
        #self.end = (end_coord[0]+transformation_vector[0], end_coord[1]+transformation_vector[1])
        transformed_star_points = np.array([(point[0] + transformation_vector[0], point[1] + transformation_vector[1]) for point in star_points])
        if ax_n:
            ax_n.plot(transformed_star_points[:, 0], transformed_star_points[:, 1], self.colour, lw=self.end_thickness)  # Plot all points as a single object
        transformed_arm_points = np.array([(point[0] + transformation_vector[0], point[1] + transformation_vector[1]) for point in star_arm_points])
        self.arm_points_array = transformed_arm_points
        self.points_array = transformed_star_points

    def mutate(self,mutation_rate=0.05):
        #(self, segment_type, start, colour, star_type, radius, arm_length, num_points, asymmetry, start_mode, end_thickness, relative_angle)
        # mutation rate chance of changing:
        self.start_mode = self.mutate_choice(self.start_mode, [StartMode.CONNECT, StartMode.JUMP], mutation_rate)
        if self.start_mode == StartMode.JUMP:
            self.start = (self.mutate_val(self.start[0], start_x_range, mutation_rate),
                          self.mutate_val(self.start[1], start_y_range, mutation_rate))

        self.end_thickness = self.mutate_val(self.end_thickness, thickness_range, mutation_rate)
        self.relative_angle = self.mutate_val(self.relative_angle, direction_range, mutation_rate)
        self.colour = self.mutate_choice(self.colour, colour_options, mutation_rate)
        self.radius = self.mutate_val(self.radius,radius_range, mutation_rate)
        self.arm_length = self.mutate_val(self.arm_length,arm_length_range, mutation_rate)

        if np.random.random() < mutation_rate:
            if self.num_points ==num_points_range[0]:
                self.num_points +=1
            elif self.num_points ==num_points_range[1]:
                self.num_points -=1
            else:
                self.num_points = self.num_points + np.random.choice([-1, 1])

        if (self.asymmetry == 0 and np.random.random() < mutation_rate) or (self.asymmetry != 0): #Less likely to mutate from 0
            self.asymmetry = self.mutate_val(self.asymmetry, asymmetry_range, mutation_rate)

        self.star_type = self.mutate_choice(self.star_type, [StarType.STRAIGHT,StarType.FLOWER,StarType.CURVED], mutation_rate)


# Factory function to create a specific segment instance and wrap it in Segment
def create_segment(start, start_mode, segment_type, end_thickness, relative_angle, colour, **kwargs):
    if segment_type == SegmentType.LINE:
        return LineSegment(
            segment_type = segment_type,
            start=start,
            start_mode=start_mode,
            end_thickness=end_thickness,
            relative_angle=relative_angle,
            colour=colour,
            start_thickness=kwargs.get('start_thickness', 1),
            length = kwargs.get('length', 1),
            curviness=kwargs.get('curviness', 0),
            curve_left=kwargs.get('curve_90', True),
            curve_location=kwargs.get('curve_location', 0.5),
            start_location = kwargs.get('start_location', 1),
            split_point = kwargs.get('split_point', 0.5)

        )
    elif segment_type == SegmentType.STAR:
        return StarSegment(
            segment_type=segment_type,
            start=start,
            start_mode=start_mode,
            end_thickness=end_thickness,
            relative_angle=relative_angle,
            colour=colour,
            radius=kwargs.get('radius', 0.5),
            arm_length=kwargs.get('arm_length', 1),
            num_points=kwargs.get('num_points', 5),
            asymmetry=kwargs.get('asymmetry', 0),
            star_type=kwargs.get('star_type', StarType.STRAIGHT),
        )
    elif segment_type == SegmentType.RANDOM_SHAPE:
        return RandomShapeSegment(
            segment_type=segment_type,
            start=start,
            start_mode=start_mode,
            end_thickness=end_thickness,
            relative_angle=relative_angle,
            colour=colour,
            bounding_size= kwargs.get('bounding_size', (1,1)),
            corners = kwargs.get('corners', []),
            lines_list = kwargs.get('lines_list', []),
        )
    else:
        raise ValueError(f"Unsupported segment type: {segment_type}")


def random_segment(eyeliner_wing = False, prev_colour=None,segment_number = 0, segment_start=(random.uniform(*start_x_range), random.uniform(*start_y_range))):
    r = random.random()
    #If we want to start with an eyeliner wing, the first two segments must be lines:
    if eyeliner_wing:
        new_segment_type = SegmentType.LINE
    else:
        if r < 0.65:
            new_segment_type = SegmentType.LINE
        elif r<0.8:
            new_segment_type = SegmentType.STAR
        else:
            new_segment_type = SegmentType.RANDOM_SHAPE

    #Pick colour, more likely to be the previous colour:
    if prev_colour is None:
        random_colour = random.choice(colour_options)
    else:
        #weights = [0.6 if colour == prev_colour else 0.1 for colour in colour_options]
        #random_colour = random.choices(colour_options, weights=weights, k=1)[0]
        random_colour = prev_colour if random.random() < 0.6 else random.choice(colour_options)

    def random_curviness(mean=0.3, stddev=0.25):
        return random_normal_within_range(mean, stddev, curviness_range)
    def random_curve_location():
        return random_normal_within_range(0.5, 0.25, relative_location_range)

    def angle_from_center(center, point):
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        return math.atan2(dy, dx)  # atan2 returns the angle in radians

    def random_lines_corners_list(n_of_corners):
        collinear = True
        # Check if points are collinear within tolerance
        while collinear:
            corners = np.array(
                [(random.uniform(*edge_initialisation_range), random.uniform(*edge_initialisation_range))
                 for i in range(n_of_corners)])
            collinear = are_points_collinear(corners)

        lines_list = [(RandomShapeLineSegment(random_curviness(0.35, 0.1),
                                              random_normal_within_range(0.5, 0.15, relative_location_range)))
                      for i in range(n_of_corners)]
        centroid = (
        sum(point[0] for point in corners) / n_of_corners, sum(point[1] for point in corners) / n_of_corners)
        sorted_corners = sorted(corners, key=lambda point: angle_from_center(centroid, point))

        return sorted_corners, lines_list

    if new_segment_type == SegmentType.LINE:
        random_start_mode = random.choice(list(StartMode))
        if segment_start == eye_corner_start and eyeliner_wing: #First segment (line) in eyeliner wing:
            random_relative_angle = random_normal_within_range(22.5, 40, direction_range)
        elif eyeliner_wing and segment_number == 1: #Second segment in eyeliner wing:
            random_relative_angle = random_normal_within_range(157.5, 40, direction_range)
            random_start_mode = StartMode.CONNECT
        elif random_start_mode == StartMode.CONNECT:
            random_relative_angle = random_from_two_distributions(135, 60, 225, 60, direction_range)
        elif random_start_mode == StartMode.CONNECT_MID:
            random_relative_angle = random_from_two_distributions(90,50,270,50, direction_range)
        elif random_start_mode == StartMode.SPLIT:
            random_relative_angle = random_from_two_distributions(90,50,270,50, direction_range)
        else:
            random_relative_angle = random.uniform(*direction_range)

        new_segment = create_segment(
            segment_type=SegmentType.LINE,
            start=segment_start,
            start_mode=random_start_mode,
            length=random.uniform(*length_range),
            relative_angle=random_relative_angle,
            start_thickness=random.uniform(*thickness_range),
            end_thickness=random.uniform(*thickness_range),
            colour=random_colour,
            curviness=0 if random.random() < 0.3 else random_curviness(),
            curve_left=random.choice([True, False]),
            curve_location=random_curve_location(),
            start_location=random_normal_within_range(0.5,0.25,relative_location_range),
            split_point=random_normal_within_range(0.5,0.25,relative_location_range)
        )
    elif new_segment_type == SegmentType.STAR:
        new_segment = create_segment(
            segment_type=SegmentType.STAR,
            start=segment_start,
            start_mode=random.choice([StartMode.CONNECT, StartMode.JUMP]),
            radius=random_normal_within_range(0.5,0.5,radius_range),
            arm_length=random_normal_within_range(0.75,0.5,arm_length_range),
            num_points=round(random_normal_within_range(5,2, num_points_range)),
            asymmetry=0 if random.random() < 0.6 else random_normal_within_range(2,2,asymmetry_range),
            star_type=random.choice([StarType.STRAIGHT, StarType.CURVED, StarType.FLOWER]),
            end_thickness=random_normal_within_range(2,2,thickness_range),
            relative_angle=random.uniform(*direction_range),
            colour=random_colour,
        )
    else:
        new_shape_overlaps = 10
        while new_shape_overlaps > max_shape_overlaps:
            n_of_corners = round(random_normal_within_range(5, 1, num_points_range))
            corners, lines = random_lines_corners_list(n_of_corners)
            new_segment = create_segment(
                segment_type=SegmentType.RANDOM_SHAPE,
                start=segment_start,
                start_mode= StartMode.CONNECT if random.random() <0.3 else StartMode.JUMP,
                end_thickness=random_normal_within_range(2, 2, thickness_range),
                relative_angle=random.uniform(*direction_range),
                colour=random_colour,
                bounding_size=(random.uniform(*random_shape_size_range), random.uniform(*random_shape_size_range)),
                corners=corners,
                lines_list=lines
            )
            prev_array = np.array([new_segment.start])
            prev_angle = 0
            prev_colour = new_segment.colour
            prev_end_thickness = new_segment.end_thickness
            new_segment.render(prev_array, prev_angle, prev_colour, prev_end_thickness)
            new_shape_overlaps = shape_overlaps(new_segment.lines_list)

    return new_segment


def set_prev_end_thickness_array(parent):
    if parent.segment_type == SegmentType.LINE:
        prev_end_thickness_array = parent.thickness_array
    else:
        prev_end_thickness_array = np.array(parent.end_thickness)

    # Check if prev_end_thickness_array is a numpy array and ensure it's iterable
    if isinstance(prev_end_thickness_array, np.ndarray):
        if prev_end_thickness_array.ndim == 0:
            # If it is a scalar (i.e., 0-dimensional), reshape it to a 1D array
            prev_end_thickness_array = prev_end_thickness_array.reshape(1)
        if prev_end_thickness_array.size == 0:
            raise ValueError("prev_end_thickness_array is empty")
    else:
        # Handle the case where it's not an array (if needed)
        print("prev_end_thickness_array:",prev_end_thickness_array)
        print("parent:",parent)
        raise ValueError("prev_end_thickness_array is not a numpy array")

    return prev_end_thickness_array

"""
# Example usage
design = EyelinerDesign()

# Adding a line segment to the design
line_segment = create_segment(
    start=(0, 0),
    start_mode=StartMode.CONNECT,
    segment_type=SegmentType.LINE,
    length=2,
    direction=45,
    start_thickness=10,
    end_thickness=1,
    color='black',
    texture='matte'
)

line_segment.mutate(0.1)
design.add_segment(line_segment)

next_start_thickness = design.get_start_thickness()
next_start = design.get_next_start_point()
curved_line_segment = create_segment(
    start=next_start,
    start_mode=StartMode.CONNECT,
    segment_type=SegmentType.LINE,
    length=1.5,
    direction=20,
    curvature=0.5,
    start_thickness=next_start_thickness,
    end_thickness=1,
    color='green',
    curviness=1,
    curve_direction=-85,
    curve_location=0.8
)
design.add_segment(curved_line_segment)

next_start = design.get_next_start_point()
next_start_thickness = design.get_start_thickness()
star_segment = create_segment(
    start=next_start,
    start_mode=StartMode.CONNECT,
    segment_type=SegmentType.STAR,
    radius=0.5,
    arm_length=1,
    num_points=4,
    asymmetry=0.5,
    curved=True,
    end_thickness=next_start_thickness
)
design.add_segment(star_segment)

# Render the design
design.render()
"""


