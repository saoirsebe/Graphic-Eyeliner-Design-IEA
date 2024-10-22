import numpy as np
import matplotlib.pyplot as plt


def rotate_point(point, center, angle_rad):
    """
    Rotate a point around a center by a given angle in radians.
    """
    # Translate point to origin
    translated_point = point - center

    # Apply rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    rotated_point = np.dot(rotation_matrix, translated_point)

    # Translate back to the original center
    return rotated_point + center


def plot_polygon(center, radius, num_sides):
    """
    Plot a regular polygon by rotating an initial line segment.
    """
    # Define the angle of rotation (in radians) between each pair of points
    angle_between_points = 2 * np.pi / num_sides

    # Define the initial two points of the line segment
    point1 = np.array([center[0] + radius, center[1]])

    # Initialize list of points with the first point
    points = [point1]

    # Generate the rest of the points by rotating the line segment
    for i in range(1, num_sides):
        # Rotate the previous point to get the next point
        next_point = rotate_point(points[-1], center, angle_between_points)
        points.append(next_point)

    # Convert the list of points to a NumPy array for easy plotting
    points = np.array(points)

    # Plot the polygon
    plt.figure(figsize=(6, 6))
    plt.plot(np.append(points[:, 0], points[0, 0]), np.append(points[:, 1], points[0, 1]), '-o')  # Close the polygon
    plt.scatter(center[0], center[1], color='red', label='Center')  # Mark the center
    plt.title(f"Regular Polygon with {num_sides} Sides")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()

    return points


# Define parameters for the pentagon
center = (0, 0)  # Center of the polygon
radius = 5  # Distance from the center to each corner
num_sides = 5  # Number of sides (pentagon)

# Plot the pentagon
pentagon_points = plot_polygon(center, radius, num_sides)

# Print the coordinates of the pentagon's points
print("Pentagon Points:\n", pentagon_points)
