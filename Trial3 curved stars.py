import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import cosine


# Function to create a quadratic Bézier curve with three control points
def bezier_curve(t, P0, P1, P2):
    return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2


# Function to create one arm of the star using a quadratic Bézier curve
def create_star_arm(center, radius, curve_control, num_points, startAngle,num_curve_points):
    centerx = center[0]
    centery = center[1]
    angle = 2 * np.pi / num_points

    P0 = np.array([centerx + radius*np.cos(startAngle)  , centery + radius*np.sin(startAngle) ])  # Inner tip of the arm
    P2 = np.array([centerx + radius*np.cos(startAngle+angle) ,centery + radius*np.sin(startAngle+angle)])  # Outer tip of the arm (on the positive x-axis)

    control_point_distance = curve_control * radius  # Controls the curvature

    M = (P0 + P2) / 2
    direction = M - center
    norm = np.linalg.norm(direction)
    if norm != 0:
        direction /= norm  # Normalize the direction vector
    P1 = M - direction * control_point_distance
    # Generate the Bézier curve points for the arm
    t_values = np.linspace(0, 1, num_curve_points)
    arm_points = np.array([bezier_curve(t, P0, P1, P2) for t in t_values])
    print(f"P0: {P0}")
    print(f"P1: {P1}")
    print(f"P2: {P2}")
    plt.scatter(center[0], center[1], color='red', label='Center')
    return arm_points


# Function to plot the star with curved edges using quadratic Bézier curves
def plot_curved_star(num_points, center, radius, curve_control, num_curve_points):
    # Set up the plot
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # Generate and plot each arm of the star using Bézier curves
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        arm_points = create_star_arm(center, radius, curve_control,num_points,angle,num_curve_points)

        # Translate the arm points to rotate around the rotation point
        translated_arm = arm_points - center
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated_arm = np.dot(translated_arm, rotation_matrix.T)
        rotated_arm += center

        plt.plot(rotated_arm[:, 0], rotated_arm[:, 1], 'b', lw=2)

    plt.title(f"{num_points}-pointed Star with Curved Arms (Quadratic Bezier)")
    plt.grid(False)
    plt.show()


# Plot a star with 5 arms, curved edges using quadratic Bézier curves
plot_curved_star(5, (0, 0), 1, 0.7, 100)
"""
plt.figure(figsize=(6, 6))
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
arm_points =create_star_arm((0,0),1,0.5,5,0)
# Plot the star arm
plt.plot(arm_points[:, 0], arm_points[:, 1], '-o')

plt.show()
"""