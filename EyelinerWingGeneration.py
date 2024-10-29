import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import random

def get_quadratic_points(a, b, c, x_start, x_end, num_points=100):
    x = np.linspace(x_start, x_end, num_points)  # Generate x values from x_start to x_end
    y = a * x ** 2 + b * x + c  # Calculate corresponding y values
    return x, y


def find_tangent_point(a, b, c, x1, y1,P):
    x = sp.symbols('x')  # Define the variable x

    # Calculate the discriminant for the quadratic equation to ensure it's real
    discriminant =  a*(a* x1**2 + b*x1 + c -y1)

    # Ensure the discriminant is non-negative
    if discriminant < 0:
        raise ValueError("The line does not intersect the quadratic at a real point.")

    sqrt_discriminant = sp.sqrt(discriminant)

    # Calculate the two possible slopes m
    m1 = 2 * a * x1 + b - (2* sqrt_discriminant)
    m2 = 2 * a * x1 + b + (2* sqrt_discriminant)

    if P==0: # Choose the slope
        m = m2
    else:
        m=m1
    line_eq = m * (x - x1) + y1  # Equation of the line with slope m passing through point (x1, y1): y = m(x - x1) + y1
    quadratic_eq = a * x ** 2 + b * x + c # Quadratic equation: y = ax^2 + bx + c

    # Find the intersection point(s) by setting the line equal to the quadratic equation
    intersection_points = sp.solve(sp.Eq(line_eq, quadratic_eq), x)

    # Since the line is tangent, it should intersect the quadratic at exactly one point
    #tangent_x = float(intersection_points[0])  # Extract the x-coordinate of the tangent point
    tangent_x = intersection_points[0].as_real_imag()[0]  # Gets the real part
    #tangent_y = float(line_eq.subs(x, tangent_x))  # Calculate the y-coordinate

    return tangent_x


def create_eyeliner_wing(length, angle, eye_corner,topStart,bottomStart):
    a=0.5
    b=0
    c=0

    angle = np.deg2rad(angle)
    e_x = eye_corner[0] + length * np.cos(angle)
    e_y = eye_corner[1] + length * np.sin(angle)
    P1 = (e_x, e_y)
    """
    # Determine P2 based on angle
    if angle > 0:
        P2 = (1, 0.5)  # Eye corner as a tuple
    else:
   """

    x_lim = find_tangent_point(-a, b, c + 1, e_x, e_y, 0)
    p0x = eye_corner[0] - topStart * (eye_corner[0] - x_lim)
    p0y = (-a) * p0x ** 2 + b * p0x + (c + 1)
    P0=(p0x, p0y)

    x_lim = find_tangent_point(a, b, c, e_x, e_y,2)
    p2x = eye_corner[0] - bottomStart * (eye_corner[0] - x_lim)
    p2y= a * p2x**2 + b * p2x + c
    P2 = (p2x, p2y)


    # Ensure P0, P1, and P2 are tuples and have consistent shape
    arm_points = np.array([P0, P1, P2])

    return arm_points


# Plotting
plt.figure(figsize=(6, 6))
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# Eye: Plotting the quadratics
x_vals, y_vals = get_quadratic_points(0.5, 0, 0, -1, 1)
plt.plot(x_vals, y_vals, label=f"$y = 0.5x^2$", color="b")
x_vals, y_vals = get_quadratic_points(-0.5, 0, 1, -1, 1)
plt.plot(x_vals, y_vals, label=f"$y = -0.5x^2 + 1$", color="b")

# Wing:
wing_points = create_eyeliner_wing(2, 10, (1,0.5),0.7,0.5)
plt.plot(wing_points[:, 0], wing_points[:, 1], 'b', lw=2)  # Plot all points as a single object
plt.grid(False)
plt.title("Eyeliner Wing")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()
