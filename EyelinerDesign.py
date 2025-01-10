import copy
import random
from logging import raiseExceptions

from conda.common.configuration import raise_errors

from AnalyseDesign import analyse_negative
from Segments import *

class EyelinerDesign:   #Creates overall design, calculates start points, renders each segment by calling their render function
    def __init__(self):
        self.segments = []
        self.n_of_lines= 0
        self.n_of_stars= 0
        self.n_of_segments= 0

    def add_segment_at(self,segment,index):
        if index < 0 or index > len(self.segments):
            raise IndexError("Index out of range. Cannot add segment at this position.")

            # Insert the segment at the specified index
        self.segments.insert(index, segment)

    def add_segment(self, segment):
        self.segments.append(segment)

    def render(self):
        """Render the eyeliner design"""
        #Branch points is a list
        branch_points = []
        fig, ax_n = plt.subplots(figsize=(3, 3))
        draw_eye_shape(ax_n)
        segment_n = 0
        prev_array = np.array([self.segments[0].start])
        prev_angle = 0
        prev_colour = self.segments[0].colour
        prev_end_thickness_array = self.segments[0].end_thickness
        for segment in self.segments:
            if segment.segment_type == SegmentType.END_POINT and len(branch_points) == 0:
                break
            elif segment.segment_type == SegmentType.BRANCH_POINT:
                segment.render(prev_array, prev_angle, prev_colour, prev_end_thickness_array)
                branch_points.append(segment)
            else:
                if segment.segment_type == SegmentType.END_POINT and len(branch_points) > 0:
                    prev_segment = branch_points[0]
                    branch_points = branch_points[1:]
                else:
                    segment.render(ax_n, prev_array, prev_angle,prev_colour,prev_end_thickness_array)

                    #Set prev_array for next segment (if segment is a branch point go back until segment isn't a branch point):
                    prev_segment =segment

                if prev_segment.segment_type == SegmentType.STAR:
                    # if segment is a star then pass in the arm points that the next segment should start at:
                    prev_array = prev_segment.arm_points_array
                else:
                     prev_array = prev_segment.points_array
                prev_angle = prev_segment.absolute_angle
                prev_colour = prev_segment.colour
                if prev_segment.segment_type == SegmentType.LINE or prev_segment.segment_type == SegmentType.BRANCH_POINT:
                    prev_end_thickness_array = prev_segment.thickness_array
                else:
                    prev_end_thickness_array = np.array(prev_segment.end_thickness)
            segment_n += 1
        return fig

    def get_start_thickness(self):
        if not self.segments:
            return 1 # Starting thickness for the first segment
        last_segment = self.segments[-1]
        return last_segment.end_thickness

    def update_design_info(self):
        self.n_of_lines = 0
        self.n_of_stars = 0
        self.n_of_segments = 0
        for segment in self.segments:
            self.n_of_segments += 1
            if segment.segment_type == SegmentType.LINE:
                self.n_of_lines += 1
            elif segment.segment_type == SegmentType.STAR:
                self.n_of_stars += 1

    def mutate_self(self,mutation_rate=0.05):
        open_branches = 1
        for current_index, segment in enumerate(self.segments):
            if segment.segment_type != SegmentType.BRANCH_POINT and segment.segment_type != SegmentType.END_POINT:
                segment.mutate(mutation_rate)
            elif segment.segment_type == SegmentType.END_POINT:
                open_branches -=1
            elif segment.segment_type == SegmentType.BRANCH_POINT:
                open_branches += 1
                #If segment is a branch point, there is a random chance it will be removed along with all the segments in that branch
                if np.random.normal() < mutation_rate/2:
                    end_point_hit = 0 #The branch that the next end point corresponds to (1 corresponds to the first open branch...)
                    next_segment_index = current_index +1
                    #Finds the index of the corresponding end point using number of open branches:
                    while (self.segments[next_segment_index].segment_type != SegmentType.END_POINT or open_branches != end_point_hit) and next_segment_index<len(self.segments)-1:
                        next_segment_index += 1
                        if self.segments[next_segment_index].segment_type == SegmentType.END_POINT:
                            end_point_hit +=1
                    if next_segment_index<len(self.segments):
                        self.segments.pop(next_segment_index)
                        #Remove all segments that where part of that branch:
                        next_segment_index -= 1
                        while self.segments[next_segment_index].segment_type != SegmentType.END_POINT:
                            self.segments.pop(next_segment_index)
                            next_segment_index -=1
                    else:
                        raise ValueError("Branch Point with No corresponding End Point")

                    self.segments.pop(current_index) #popping one after?!

            current_index +=1

        # Random chance of adding in a new segment (Not a branch point or end point):
        if np.random.normal() < mutation_rate:
            new_segment = random_segment(False)
            self.add_segment_at(new_segment, np.random.randint(0, len(self.segments)))

        #Random chance of swapping the location of a segment (Not a branch point or end point):
        if np.random.normal() < mutation_rate:
            to_change = np.random.randint(0, len(self.segments) - 1)
            while self.segments[to_change].segment_type == SegmentType.END_POINT or self.segments[to_change].segment_type == SegmentType.BRANCH_POINT or (self.segments[to_change-1].segment_type == SegmentType.END_POINT and self.segments[to_change+1].segment_type == SegmentType.END_POINT):
                to_change = np.random.randint(0, len(self.segments) - 1)
            new_position = np.random.randint(0, len(self.segments) - 1)
            while new_position == to_change:
                new_position = np.random.randint(0, len(self.segments))
            segment = self.segments.pop(to_change)
            self.segments.insert(new_position, segment)
        return self

    def mutate(self,mutation_rate=0.05):
        new_gene = copy.deepcopy(self)
        new_gene= new_gene.mutate_self(mutation_rate)
        fig=new_gene.render()
        plt.close(fig)
        overlap_score = analyse_negative(new_gene)
        print("overlap_score", overlap_score)
        while overlap_score<=min_fitness_score:
            new_gene = copy.deepcopy(self)
            new_gene = new_gene.mutate_self(mutation_rate)
            fig = new_gene.render()
            plt.close(fig)
            overlap_score = analyse_negative(new_gene)
            print("overlap_score", overlap_score)
        return new_gene

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

def random_segment(can_branch = False,segment_start=(random.uniform(*start_x_range), random.uniform(*start_y_range)),eyeliner_wing = False, segment_number = 0):
    r = random.random()
    if eyeliner_wing and (segment_number == 0 or segment_number == 2):
        new_segment_type = SegmentType.LINE
    elif eyeliner_wing and segment_number == 1:
        new_segment_type = SegmentType.BRANCH_POINT
    elif can_branch:
        if r < 0.5:
            new_segment_type = SegmentType.LINE
        elif r < 0.7:
            new_segment_type = SegmentType.STAR
        else:
            new_segment_type = SegmentType.BRANCH_POINT
    else:
        if r < 0.6:
            new_segment_type = SegmentType.LINE
        else:
            new_segment_type = SegmentType.STAR

    if new_segment_type == SegmentType.LINE:
        random_start_mode = random.choice(list(StartMode))
        if segment_start == (3, 1.5) and eyeliner_wing: #First segment (line) in eyeliner wing:
            random_relative_angle = random_normal_within_range(45, 50, direction_range)
        elif eyeliner_wing and segment_number == 2: #Third segment (second line) in eyeliner wing:
            random_relative_angle = random_normal_within_range(150, 50, direction_range)
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
            colour=random.choice(colour_options),
            curviness=0 if random.random() < 0.5 else random_normal_within_range(0.1,0.5,curviness_range),
            curve_direction=random_from_two_distributions(90,40,270,40, direction_range),
            curve_location=random_normal_within_range(0.5,0.25,relative_location_range),
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
            colour=random.choice(colour_options),
        )
    elif new_segment_type == SegmentType.BRANCH_POINT:
        new_segment = BranchPointSegment()
    return new_segment

def random_gene(gene_n):
    design = EyelinerDesign()
    if (initial_gene_pool_size/3 < gene_n <= 2* (initial_gene_pool_size/3)):
        n_objects = 3
    else:
        n_objects = round(random_normal_within_range(6,5,(1, 10)))
    n_of_branches = 1
    first_branch = True
    while n_of_branches >0:
        n_of_branches -=1
        can_branch = False #Cant branch if first segment in a new branch
        for i in range(n_objects):
            #The first 2 thirds of the initial population start at the corner of the eye:
            if first_branch and(gene_n <= 2 *(initial_gene_pool_size/3)) and i==0:
                segment_start = (3, 1.5)
            else:
                segment_start = (random.uniform(*start_x_range), random.uniform(*start_y_range))

            #If eyeliner_wing (second third of the initial population) then the second segment is likely to be a line directed (back to the eye) in the shape of an eyeliner wing:
            if first_branch and initial_gene_pool_size/3 < gene_n <= 2* (initial_gene_pool_size / 3) and i <= 2:
                new_segment = random_segment(can_branch,segment_start,True,i)
                print("Design number:", gene_n)
                print("new_segment", new_segment)
            else:
                new_segment = random_segment(can_branch, segment_start)

            if new_segment.segment_type == SegmentType.BRANCH_POINT:
                n_of_branches +=1

            design.add_segment(new_segment)
            can_branch = True

        first_branch = False

        design.add_segment(EndPointSegment())
        n_objects = round(random_normal_within_range(2,3,(1, 10)))

    #design.add_segment(EndPointSegment())
    #design.update_design_info()
    return design

"""
random_design = random_gene(0)
fig = random_design.render()
fig.show()
score = analyse_design_shapes(random_design)
print("Score:", score)
"""