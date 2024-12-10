import copy
import random
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
        fig, ax_n = plt.subplots(figsize=(3, 3))
        """Render the eyeliner design using matplotlib."""
        draw_eye_shape(ax_n)
        segment_n = 0
        prev_array = np.array([self.segments[0].start])
        prev_angle = 0
        prev_colour = self.segments[0].colour
        prev_end_thickness_array = self.segments[0].end_thickness
        for segment in self.segments:
            segment.render(ax_n, prev_array, prev_angle,prev_colour,prev_end_thickness_array)
            if segment.segment_type == SegmentType.STAR:
                prev_array = self.segments[segment_n].arm_points_array #if previous segment was a star then pass in the arm points that the next segment should start at
            else:
                prev_array = self.segments[segment_n].points_array
            prev_angle = self.segments[segment_n].absolute_angle
            prev_colour = self.segments[segment_n].colour
            if segment.segment_type == SegmentType.LINE:
                prev_end_thickness_array = self.segments[segment_n].thickness_array
            else:
                prev_end_thickness_array = np.array(self.segments[segment_n].end_thickness)
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

    def mutate(self,mutation_rate=0.1):
        new_gene = copy.deepcopy(self)
        for segment in new_gene.segments:
            segment.mutate(mutation_rate)
        if np.random.normal()< mutation_rate:
            new_segment = random_segment()
            self.add_segment_at(new_segment,np.random.randint(0,len(self.segments)))
        if np.random.normal()< mutation_rate:
            to_change = np.random.randint(0,len(self.segments)-1)
            new_position = np.random.randint(0,len(self.segments)-1)
            while new_position == to_change:
                new_position = np.random.randint(0, len(self.segments))
            segment = self.segments.pop(to_change)
            self.segments.insert(new_position, segment)
        return new_gene

def random_segment(segment_start=(random.uniform(*start_x_range), random.uniform(*start_y_range))):
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
            star_type=random.choice([StarType.STRAIGHT, StarType.CURVED, StarType.FLOWER]),
            end_thickness=random.uniform(*thickness_range),
            relative_angle=random.uniform(*direction_range),
            colour=random.choice(colour_options),
        )
    return new_segment

def random_gene(gene_n):
    design = EyelinerDesign()
    n_objects = random.randint(2, 10)
    for i in range(n_objects):
        if (gene_n ==0 or gene_n ==1 or gene_n == 2) and i==0:
            segment_start = (3, 1.5)
        else:
            segment_start = (random.uniform(*start_x_range), random.uniform(*start_y_range))

        new_segment = random_segment(segment_start)

        design.add_segment(new_segment)
        next_start_thickness = design.get_start_thickness()
    design.update_design_info()
    return design