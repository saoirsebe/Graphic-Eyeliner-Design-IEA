import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import comb

def bezier_curve(t, *P):
    """
    Calculate a point on a Bézier curve using a set of control points.

    Parameters:
    t (float): Parameter ranging from 0 to 1.
    P (tuple): Control points (2D points as tuples or arrays).

    Returns:
    np.array: A point on the Bézier curve at parameter t.
    """
    n = len(P) - 1
    curve_point = sum(comb(n, i) * (1 - t) ** (n - i) * t ** i * np.array(P[i]) for i in range(n + 1))
    return curve_point

def create_bezier_curve(length, number_of_control_points):
    """
    Create a Bézier curve using a specified length and number of control points.

    Parameters:
    length (float): Total length of the curve along the x-axis.
    number_of_control_points (int): Number of control points to generate.

    Returns:
    np.array: Points along the Bézier curve.
    """
    # Generate control points along the x-axis with small random variations in y for wiggliness
    control_points = []
    for i in range(number_of_control_points):
        x_pos = i * (length / (number_of_control_points - 1))  # Evenly spaced x-coordinates
        y_pos = np.random.uniform(-0.5, 0.5)  # Random y variations for wiggles
        control_points.append((x_pos, y_pos))

    # Generate Bézier curve points using the control points
    t_values = np.linspace(0, 1, 100)
    curve_points = np.array([bezier_curve(t, *control_points) for t in t_values])

    return curve_points, control_points

def create_line(length, is_straight, nOfWiggles):
    t_values = np.linspace(0, 1, 100)
    curve_points = np.array([bezier_curve(t, nOfWiggles) for t in t_values])
    return curve_points


# Set axis limits and labels
plt.figure(figsize=(6, 6))
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
line_points = create_line(10, True, 5)
plt.plot(line_points[:, 0], line_points[:, 1], 'b', lw=2)
plt.grid(False)
plt.title("Eyeliner Wing")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()

