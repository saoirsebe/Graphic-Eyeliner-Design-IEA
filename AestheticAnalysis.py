from matplotlib import pyplot as plt

from A import SegmentType, StartMode
from EyelinerWingGeneration import get_quadratic_points, generate_eye_curve_directions
import numpy as np
from scipy.interpolate import interp1d


def compare_curves(bezier_points, eye_points, eye_curve_shape, is_upper,num_samples=100):
    """
    Compare the curvature and direction of a Bezier curve and a quadratic curve.

    Returns:
        dict: A dictionary containing shape similarity, direction similarity, and overall similarity.
    """

    def resample_curve(points, num_resize_val):
        """Resample a curve to have exactly num_samples points."""
        distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
        total_distance = cumulative_distances[-1]
        uniform_distances = np.linspace(0, total_distance, num_resize_val)
        interp_x = interp1d(cumulative_distances, points[:, 0], kind='linear')
        interp_y = interp1d(cumulative_distances, points[:, 1], kind='linear')
        return np.vstack((interp_x(uniform_distances), interp_y(uniform_distances))).T

    def resample_directions_or_curvatures(values, points, num_resize_val):
        """Resample directions or curvatures arrays based on the distance between points."""
        # Compute distances between successive points
        distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
        cumulative_distances = np.insert(np.cumsum(distances), 0,0)  # Insert a 0 at the start to match the number of 'values'

        # Create uniform distances for resampling
        total_distance = cumulative_distances[-1]
        uniform_distances = np.linspace(0, total_distance, num_resize_val)
        # Interpolate values (curvatures or directions) based on cumulative distances
        if len(values)==len(cumulative_distances) - 2:
            cumulative_distances = cumulative_distances[1:-1]
        elif len(values)==len(cumulative_distances) - 1:
            cumulative_distances=cumulative_distances[1:]
        else:
            raise ValueError("The length of 'values' must be one/two less than the length of 'points'")
        #cumulative_distances_unique = cumulative_distances[0]
        cumulative_distances_unique, unique_indices = np.unique(cumulative_distances, return_index=True)
        values_unique = values[unique_indices]
        if cumulative_distances_unique.size == 0:
            print("cumulative_distances:", cumulative_distances)
            print("distances:", distances)
            print("points:", points)
            print("values:", values)
            raise ValueError("cumulative_distances_unique array must not be empty.")
        if values_unique.size == 0:
            print("values:", values)
            raise ValueError("values_unique array must not be empty.")
        if cumulative_distances_unique.shape[0] != values_unique.shape[0]:
            raise ValueError("Input arrays must have the same length.")

        interp_values = interp1d(cumulative_distances_unique, values_unique, kind='linear',axis=0, bounds_error=False,fill_value="extrapolate")
        #print("NaN in cumulative_distances:", np.isnan(cumulative_distances).any())

        #print("Cumulative Distances Range:", min(cumulative_distances), max(cumulative_distances))
        #print("Uniform Distances Range:", min(uniform_distances), max(uniform_distances))
        #print(not np.all(np.diff(cumulative_distances) > 0))
        #extrapolated_values = interp_values([min(cumulative_distances) - 1, max(cumulative_distances) + 1])
        #print(extrapolated_values)

        return interp_values(uniform_distances)


    def apply_upper_lower_condition(bezier_curve, quadratic_curve, is_upper, threshold=0.9):
        count_valid = 0
        total_points = len(bezier_curve)

        for bezier_point, quadratic_point in zip(bezier_curve, quadratic_curve):
            if is_upper:
                # Bezier should be above the quadratic curve
                if bezier_point[1] > quadratic_point[1]:
                    count_valid += 1
            else:
                # Bezier should be below the quadratic curve
                if bezier_point[1] < quadratic_point[1]:
                    count_valid += 1

        # If 90% or more of the points are valid (either above or below), return 1, otherwise return 0
        if count_valid / total_points >= threshold:
            return 1
        else:
            return 0

    # Resample both curves to have the same number of points
    num_resize = max(len(bezier_points) , len(eye_points))
    bezier_position_resampled = resample_curve(bezier_points, num_resize)
    eye_position_resampled = resample_curve(eye_points, num_resize)

    position_score = apply_upper_lower_condition(bezier_position_resampled, eye_position_resampled, is_upper)
    if position_score == 0:
        return{
            "shape_similarity": 0,
            "direction_similarity": 0,
            "overall_similarity": 0
        }
    def direction_between_points(point1,point2):
        tangent = np.diff((point1, point2), axis=0)
        # Prevent division by zero when normalizing
        norms = np.linalg.norm(tangent, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8  # Prevent zero vectors by setting a small value
        return tangent / norms

    def compute_directions_new(points):
        directions = []
        for i in range(len(points)-2):
            val = i
            next_val = i +1
            distance_from_next_point = np.linalg.norm(np.array(points[val]) - np.array(points[next_val]))
            while distance_from_next_point <= 0.0001:
                next_val +=1
                if next_val <len(points):
                    distance_from_next_point = np.linalg.norm(np.array(points[val]) - np.array(points[next_val]))
                else:
                    val = i-1
                    next_val =i
                    while distance_from_next_point <= 0.0001:
                        val -= 1
                        distance_from_next_point = np.linalg.norm(np.array(points[val]) - np.array(next_val))
            directions.append(direction_between_points(points[val], points[next_val]))
        return np.array(directions)

    # Compute tangent vectors and normalize to get directions
    def compute_directions(points):
        unique_points = [points[0]]  # Always keep the first point
        duplicate_indices = []
        for i in range(1, len(points)-1):
            if not np.array_equal(points[i], points[i - 1]):
                unique_points.append(points[i])
            else:
                duplicate_indices.append(i)
        unique_points = np.array(unique_points)

        if len(unique_points)<=1:
            return np.array([])

        tangents = np.diff(unique_points, axis=0)
        # Prevent division by zero when normalizing
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8  # Prevent zero vectors by setting a small value
        directions = tangents / norms
        if len(directions) == 0:
            print("points:", points)
            raise ValueError("len(directions) == 0")

        directions_with_duplicates = []
        directions_idx = 0  # Index to track unique direction array
        for i in range(len(points)-2):
            if np.array_equal(points[i], points[i + 1]) and directions_idx!=0:
                directions_with_duplicates.append(directions[directions_idx-1])
            elif np.array_equal(points[i], points[i + 1]) and directions_idx==0:
                directions_with_duplicates.append(directions[directions_idx])
            else:
                # For duplicates, assign the same direction as the last unique point
                directions_with_duplicates.append(directions[directions_idx])
                directions_idx += 1

        return np.array(directions_with_duplicates)

    # Compute curvature for both sets of points
    def calculate_curvature(points):
        vectors = np.diff(points, axis=0)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        curvatures = np.diff(angles)
        return curvatures

    def get_overlapping_points(curve1, curve2, tolerance=0.1):
        overlapping_points_curve1 = []
        overlapping_points_curve2 = []

        for point1 in curve1:
            # Find points in curve2 with x-value within the tolerance range
            mask = np.abs(curve2[:, 0] - point1[0]) <= tolerance
            if np.any(mask):
                # Get the corresponding point(s) from curve2
                overlapping_points_curve2.append(curve2[mask][0])  # Just take the first match
                overlapping_points_curve1.append(point1)

        return np.array(overlapping_points_curve1), np.array(overlapping_points_curve2)

    overlapping_points_segment, overlapping_points_eye_shape = get_overlapping_points(bezier_points, eye_curve_shape)

    if overlapping_points_segment.shape[0]>4 and overlapping_points_eye_shape.shape[0]>4:
        """
        # Plot the curves
        plt.figure(figsize=(8, 6))
        plt.plot(overlapping_points_bezier[:, 0], overlapping_points_bezier[:, 1], label='Bezier Curve', color='b')
        plt.plot(overlapping_points_eye_shape[:, 0], overlapping_points_eye_shape[:, 1], label=f'Quadratic Curve {is_upper}',color='g')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')  # Equal aspect ratio ensures that arrows are displayed proportionally
        # Show plot
        plt.show()
        """
        bezier_curvature = calculate_curvature(overlapping_points_segment)
        eye_curvature = calculate_curvature(overlapping_points_eye_shape)
        # Compute shape similarity (curvature comparison)
        num_resize = max(len(overlapping_points_segment), len(overlapping_points_eye_shape))
        if len(bezier_curvature) <=1:
            print("len(bezier_curvature) <=1:")
            print("overlapping_points_bezier:", overlapping_points_segment)
            print("bezier_points", bezier_points)
            print("eye_curve_shape",eye_curve_shape)
            shape_similarity = 0
        else:
            bezier_curvature_resampled = resample_directions_or_curvatures(bezier_curvature,overlapping_points_segment, num_resize)
            eye_curvature_resampled = resample_directions_or_curvatures(eye_curvature, overlapping_points_eye_shape,num_resize)
            #shape_similarity = 1 - np.mean(np.abs(bezier_curvature - eye_curvature))
            shape_similarity = 1 - np.sqrt(np.mean((bezier_curvature_resampled - eye_curvature_resampled) ** 2))

        segment_directions = compute_directions_new(overlapping_points_segment)
        eye_curve_directions = compute_directions_new(overlapping_points_eye_shape)

        if segment_directions.size == 0:
            print("bezier_directions.size == 0")
            print("overlapping_points_bezier",overlapping_points_segment)
            direction_similarity = 0
        else:
            #print("eye_curve_directions",eye_curve_directions)
            num_resize = max(len(overlapping_points_segment) , len(overlapping_points_eye_shape))
            bezier_directions_resampled = resample_directions_or_curvatures(segment_directions,overlapping_points_segment, num_resize)
            eye_directions_resampled = resample_directions_or_curvatures(eye_curve_directions,overlapping_points_eye_shape, num_resize)
            #print("eye_directions_resampled",eye_directions_resampled)
            # Compute direction similarity (angle between normalized tangent vectors)
            dot_products = np.sum(bezier_directions_resampled * eye_directions_resampled, axis=1)
            direction_similarity = np.mean(dot_products)
    else:
        shape_similarity = 0
        direction_similarity = 0

    # Combine into overall similarity
    overall_similarity = (shape_similarity + direction_similarity) / 2
    return {
        "shape_similarity": shape_similarity,
        "direction_similarity": direction_similarity,
        "overall_similarity": overall_similarity
    }


def score_segment(segment, upper_curve, lower_curve, tolerance=0.1):
    """
    Score a single segment based on how well it aligns with the natural curves.
    """
    points = segment.points_array
    # Calculate curvature of the segment
    top_eye_curve, bottom_eye_curve = generate_eye_curve_directions()
    upper_curve_results= compare_curves(points,upper_curve,top_eye_curve,True, num_samples=100)
    lower_curve_results= compare_curves(points,lower_curve,bottom_eye_curve,False, num_samples=100)
    #print("upper_curve_match", upper_curve_results)
    #print("lower_curve_match", lower_curve_results)

    # Assign scores
    score = 0
    if upper_curve_results["overall_similarity"]>0.6:
        score+= 10*upper_curve_results["overall_similarity"]
    elif lower_curve_results["overall_similarity"]>0.6:
        score += 10* lower_curve_results["overall_similarity"]
    #else:
    #    score -= 2  # Penalty for deviating

    return score


def analyse_design_shapes(design):
    """
    Analyse the entire design and calculate a total score.
    """
    upper_x, upper_y = get_quadratic_points(-0.5, 0, 1, -1, 1)
    lower_x, lower_y = get_quadratic_points(0.5, 0, 0, -1, 1)
    upper_curve = np.column_stack(([x * 3 for x in upper_x], [y * 3 for y in upper_y]))
    lower_curve = np.column_stack(([x * 3 for x in lower_x], [y * 3 for y in lower_y]))

    # Define wing direction
    wing_direction = [1, 0.2]

    total_score = 0
    for segment in design.segments:
        if segment.segment_type == SegmentType.LINE:
            total_score += score_segment(segment, upper_curve, lower_curve)
    return total_score
"""
random_design = random_gene(0)

random_design = EyelinerDesign()
new_segment = create_segment(
        segment_type=SegmentType.LINE,
        start = (0,3),
        start_mode=StartMode.CONNECT,
        length=3,
        relative_angle=0,
        start_thickness=2.5,
        end_thickness=1,
        colour="red",
        curviness= 0.5 ,
        curve_direction=0.5,
        curve_location=0.5,
        start_location=0.6,
        split_point=0.2

)
random_design.add_segment(new_segment)
new_segment = create_segment(
        segment_type=SegmentType.LINE,
        start = (1,-1.5),
        start_mode=StartMode.JUMP,
        length=1.5,
        relative_angle=110,
        start_thickness=6,
        end_thickness=4,
        colour="orange",
        curviness= 0.5 ,
        curve_direction=20,
        curve_location=0.5,
        start_location=0.6,
        split_point=0

)
random_design.add_segment(new_segment)
new_segment = create_segment(
        segment_type=SegmentType.LINE,
        start = (-2,3),
        start_mode=StartMode.JUMP,
        length=7,
        relative_angle=70,
        start_thickness=2.5,
        end_thickness=1,
        colour="pink",
        curviness= 0 ,
        curve_direction=90,
        curve_location=0.5,
        start_location=0.6,
        split_point=0

)
random_design.add_segment(new_segment)

fig = random_design.render()
fig.show()
score = analyse_design_shapes(random_design)
print("Score:", score)
"""