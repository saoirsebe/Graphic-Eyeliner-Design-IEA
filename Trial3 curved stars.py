import numpy as np
import matplotlib.pyplot as plt


# Function to create a quadratic bezier curve with three control points
def bezier_curve(t, P0, P1, P2):
    return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2


# Function to create one arm of the star using a quadratic Bézier curve
def create_star_arm(center, radius, curve_control, arm_length, num_points):
    """
    Create one arm of the star using a quadratic Bézier curve.

    Parameters:
    - center: The center of the star.
    - radius: Length from the center to the outer tip of the star.
    - curve_control: Controls how much the arm curves (0 is straight, 1 is highly curved).
    - arm_length: Length of the inner part of the arm (towards the center).
    - num_points: Number of points to use for the curve.

    Returns:
    - Array of points representing the star arm.
    """
    # Define the start point (P0) at the center of the star
    P0 = np.array([center[0], center[1]])  # Center point (fix here)

    # Define the end point (P2) at the tip of the star arm
    P2 = np.array([radius * np.cos(0), radius * np.sin(0)])  # Outer tip of the arm (on the positive x-axis)

    # Control point (P1) to create the curve
    control_point_distance = curve_control * radius  # Controls the curvature

    # First control point (P1)
    P1 = np.array([P0[0] + control_point_distance * np.cos(np.pi / 4), P0[1] + control_point_distance * np.sin(np.pi / 4)])

    # Generate the Bézier curve points for the arm
    t_values = np.linspace(0, 1, num_points)
    arm_points = np.array([bezier_curve(t, P0, P1, P2) for t in t_values])

    return arm_points


# Function to plot the star with curved edges using quadratic Bézier curves
def plot_curved_star(num_points, center, radius, curve_control, arm_length, num_curve_points):

    # Set up the plot
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # Generate and plot each arm of the star using Bézier curves
    for i in range(num_points):
        # Rotate the control points to form the star shape
        angle = 2 * np.pi * i / num_points

        # Create the arm points using quadratic Bézier curve
        arm_points = create_star_arm(center, radius, curve_control, arm_length, num_curve_points)

        # Rotate the arm points by the angle
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated_arm = np.dot(arm_points, rotation_matrix.T)

        # Plot the curved star arm
        plt.plot(rotated_arm[:, 0], rotated_arm[:, 1], 'b', lw=2)

    plt.title(f"{num_points}-pointed Star with Curved Arms (Quadratic Bezier)")
    plt.grid(False)
    plt.show()


# Plot a star with 5 arms, curved edges using quadratic Bézier curves
plot_curved_star(6, (0, 0), 1.0, 0.7, 0.5, 100)
