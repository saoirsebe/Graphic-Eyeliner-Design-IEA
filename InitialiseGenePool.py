import matplotlib.pyplot as plt

from A import min_fitness_score, initial_gene_pool_size
from AnalyseDesign import analyse_negative, analyse_positive
from EyelinerDesign import random_gene


def initialise_gene_pool():
    gene_pool = [random_gene(i) for i in range(initial_gene_pool_size)]
    scored_genes = []
    # Check the overlap_score of each and re-generate if lower than min_fitness_score:
    for idx, gene in enumerate(gene_pool):
        fig = gene.render()  # Render each gene on its specific subplot
        plt.close(fig)
        overlap_score = analyse_negative(gene)
        #print("first overlap_score: ", overlap_score)
        while overlap_score <= min_fitness_score:
            gene_pool[idx] = random_gene(idx)
            gene = gene_pool[idx]  # Update the loop variable with the new gene
            fig = gene.render()  # Render the new gene
            plt.close(fig)
            overlap_score = analyse_negative(gene)
            #print("New overlap_score: ", overlap_score)

        #print("final overlap_score: ", overlap_score)
        scored_genes.append((gene,  + analyse_positive(gene))) #overlap_score +

    scored_genes.sort(key=lambda x: x[1], reverse=True)  # Sort by the gene score
    print("Gene score: ", scored_genes)

    gene_pool = [gene for gene, score in scored_genes[:6]]

    return gene_pool


"""
num_points=random.randint(*num_points_range)
new_star_segment = create_segment(
                segment_type=SegmentType.STAR,
                start = (0,0),
                start_mode=StartMode.CONNECT,
                radius=random.uniform(*radius_range),
                arm_length=2,#random.uniform(*arm_length_range),
                num_points=num_points,
                asymmetry= 0,#random.uniform(*asymmetry_range),
                curved=random.choice([True, False]),
                end_thickness=random.uniform(*thickness_range),
                relative_angle = random.uniform(*direction_range),
                end_arm= random.uniform(0,num_points-1)
            )
design.add_segment(new_star_segment)
next_start_thickness = design.get_start_thickness()



for i in range(5):
    new_segment = create_segment(
        segment_type=SegmentType.LINE,
        start = (0,0),
        start_mode=random.choice([StartMode.CONNECT, StartMode.SPLIT, StartMode.CONNECT_MID]),
        length=random.uniform(*length_range),
        relative_angle=random.uniform(*direction_range),
        start_thickness=next_start_thickness,
        end_thickness=random.uniform(*thickness_range),
        color=random.choice(colour_options),
        curviness= 0 if random.random() < 0.5 else random.uniform(*curviness_range),
        curve_direction=random.uniform(*curve_direction_range),
        curve_location=random.uniform(*curve_location_range),
        start_location=random.uniform(*curve_location_range),
        split_point=random.uniform(*curve_location_range)

    )
    print("start mode:", new_segment.start_mode)
    design.add_segment(new_segment)
    next_start_thickness = design.get_start_thickness()
"""
"""
design = EyelinerDesign()
new_segment = create_segment(
        segment_type=SegmentType.LINE,
        start = (3,1.5),
        start_mode=StartMode.CONNECT,
        length=3,
        relative_angle=40,
        start_thickness=2.5,
        end_thickness=1,
        colour="red",
        curviness= 0.2 ,
        curve_direction=0.2,
        curve_location=0.5,
        start_location=0.6,
        split_point=0.2

)
design.add_segment(new_segment)
new_segment = create_segment(
        segment_type=SegmentType.LINE,
        start = (3,1.5),
        start_mode=StartMode.CONNECT_MID,
        length=1.5,
        relative_angle=110,
        start_thickness=6,
        end_thickness=4,
        colour="orange",
        curviness= 0 ,
        curve_direction=0.2,
        curve_location=0.5,
        start_location=0.6,
        split_point=0.2

)
design.add_segment(new_segment)
new_segment = create_segment(
        segment_type=SegmentType.LINE,
        start = (3,1.5),
        start_mode=StartMode.SPLIT,
        length=7,
        relative_angle=20,
        start_thickness=2.5,
        end_thickness=1,
        colour="pink",
        curviness= 0 ,
        curve_direction=0,
        curve_location=0.5,
        start_location=0.6,
        split_point=0.2

)
design.add_segment(new_segment)
fig = design.render()
fig.show()


new_segment = create_segment(
        segment_type=SegmentType.LINE,
        start = (3,1.5),
        start_mode=StartMode.CONNECT,
        length=random.uniform(*length_range),
        relative_angle=random.uniform(*direction_range),
        start_thickness=next_start_thickness,
        end_thickness=random.uniform(*thickness_range),
        colour="red",
        curviness= 0 if random.random() < 0.5 else random.uniform(*curviness_range),
        curve_direction=random.uniform(*curve_direction_range),
        curve_location=random.uniform(*curve_location_range),
        start_location=random.uniform(*curve_location_range),
        split_point=random.uniform(*curve_location_range)

)
design.add_segment(new_segment)
next_start_thickness = design.get_start_thickness()

new_segment = create_segment(
        segment_type=SegmentType.LINE,
        start = (3,1.5),
        start_mode=StartMode.CONNECT_MID,
        length=random.uniform(*length_range),
        relative_angle=random.uniform(*direction_range),
        start_thickness=next_start_thickness,
        end_thickness=random.uniform(*thickness_range),
        colour="green",
        curviness= 0 if random.random() < 0.5 else random.uniform(*curviness_range),
        curve_direction=random.uniform(*curve_direction_range),
        curve_location=random.uniform(*curve_location_range),
        start_location=random.uniform(*curve_location_range),
        split_point=random.uniform(*curve_location_range)

)
design.add_segment(new_segment)
next_start_thickness = design.get_start_thickness()

new_segment = create_segment(
        segment_type=SegmentType.LINE,
        start = (0,0),
        start_mode=StartMode.SPLIT,
        length=random.uniform(*length_range),
        relative_angle=random.uniform(*direction_range),
        start_thickness=next_start_thickness,
        end_thickness=random.uniform(*thickness_range),
        colour="blue",
        curviness= 0 if random.random() < 0.5 else random.uniform(*curviness_range),
        curve_direction=random.uniform(*curve_direction_range),
        curve_location=random.uniform(*curve_location_range),
        start_location=random.uniform(*curve_location_range),
        split_point=random.uniform(*curve_location_range)

)
design.add_segment(new_segment)
"""
"""
line_seg = LineSegment(SegmentType.LINE,(0,0),StartMode.JUMP,5,230,2,4,"purple",0,0,0.8,0,0)
#design.render()
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_aspect('equal')
line_seg.render(ax,[],0,"white",3)
plt.show()


star_seg = StarSegment(SegmentType.STAR,(0,0),"pink",StarType.FLOWER,1,2,5,0,StartMode.JUMP,3,30)
# curviness of star FIX!!!!!!!!!
#Asymetry broken FIX!!!
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_aspect('equal')
star_seg.render(ax,[],0,"white",3)
plt.show()
"""
