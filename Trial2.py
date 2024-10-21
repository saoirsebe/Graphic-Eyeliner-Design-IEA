import numpy as np
import matplotlib.pyplot as plt


# Define the arrow tip unit set (as a 3-point polygon)
def create_arrow_tip():
    # Arrow tip coordinates (relative to the center)
    arrow_tip = np.array([
        [0, 1],  # Tip of the arrow (pointed upwards)
        [-0.5, -0.5],  # Bottom-left of the arrow
        [0.5, -0.5]  # Bottom-right of the arrow
    ])
    return arrow_tip


# Define the rotation matrix for a given angle (in degrees)
def rotation_matrix(theta):
    """ Return the 2D rotation matrix for an angle theta in degrees """
    theta_rad = np.deg2rad(theta)
    return np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])


# Apply the rotation matrix to a set of points
def rotate_points(points, theta):
    """ Rotate a set of 2D points by theta degrees """
    rot_matrix = rotation_matrix(theta)
    return np.dot(points, rot_matrix.T)


# Plot the star with the arrow tip shape
def plot_star(num_points=5, arm_length=1.0, arrow_tip_size=0.5):
    # Create the base arrow tip unit set (one arm of the star)
    arrow_tip = create_arrow_tip() * arrow_tip_size

    # Set up the plot
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # Draw the star arms by rotating the arrow tip around the center
    for i in range(num_points):
        # Calculate the rotation angle for each arm
        angle = 360 / num_points * i

        # Rotate the arrow tip to its correct position
        rotated_arrow = rotate_points(arrow_tip, angle)

        # Plot the rotated arrow tip
        # Add the arrow to the plot, centered at the origin (0, 0)
        plt.fill(rotated_arrow[:, 0], rotated_arrow[:, 1], 'b', edgecolor='black')

    plt.title(f"{num_points}-pointed Star with Arrow Tip")
    plt.grid(False)
    plt.show()


# Plot a star with 5 arms and an arrow tip shape
plot_star(num_points=5, arm_length=1.0, arrow_tip_size=0.5)
