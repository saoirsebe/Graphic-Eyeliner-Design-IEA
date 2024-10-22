import numpy as np
import matplotlib.pyplot as plt

# Function to create a quadratic bezier curve with three control points
def bezier_curve(t, P0, P1, P2):
    return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2


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
    # Define the start point (P0) at the center of the star, moved inward by arm_length
    P0 = np.array([center[0] + arm_length * np.cos(0), center[1] + arm_length * np.sin(0)])
    # Define the end point (P2) at the tip of the star arm
    P2 = np.array([center[0] + radius * np.cos(0), center[1] + radius * np.sin(0)])

    # Control point (P1) to create the curve
    control_distance = curve_control * np.linalg.norm(P2 - P0)  # Determines how far from P0 to place the control point

    # Find the midpoint between P0 and P2 and move the control point upwards
    midpoint = (P0 + P2) / 2
    P1 = np.array([midpoint[0], midpoint[1] + control_distance])  # Adjust height based on control_distance

    # Generate the Bézier curve points for the arm
    t_values = np.linspace(0, 1, num_points)
    arm_points = np.array([bezier_curve(t, P0, P1, P2) for t in t_values])

    return arm_points


def plot_curved_star(num_points, center, radius, curve_control, arm_length, num_curve_points):
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # Generate and plot each arm of the star using bezier curves
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        arm_points = create_star_arm(center=center, radius=radius, curve_control=curve_control, arm_length=arm_length, num_points=num_curve_points)

        # Define a rotation point (e.g., the center of the star)
        rotation_point = center  # You can change this to any other point

        # Translate the arm points to rotate around the rotation point
        translated_arm = arm_points - rotation_point

        # Rotation matrix for rotating around the origin
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        # Rotate the translated arm points
        rotated_arm = np.dot(translated_arm, rotation_matrix.T)

        # Translate back to the original position
        rotated_arm += rotation_point

        plt.plot(rotated_arm[:, 0], rotated_arm[:, 1], 'b', lw=2)

    plt.title(f"{num_points}-pointed Star with Curved Arms (Quadratic Bezier)")
    plt.grid(False)
    plt.show()

# Plot a star with 5 arms, curved edges using quadratic bezier curves
plot_curved_star(5, (0, 0), 1.0, 0.5, 0.5, 100)
