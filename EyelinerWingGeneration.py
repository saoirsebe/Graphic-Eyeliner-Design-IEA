import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import random

from Trial import bezier_curve

def bezier_curve(t, P0, P1, P2):
    return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2

def get_quadratic_points(a, b, c, x_start, x_end, num_points=100):
    x = np.linspace(x_start, x_end, num_points)  # Generate x values from x_start to x_end
    y = a * x ** 2 + b * x + c  # Calculate corresponding y values
    return x, y

def vector_direction(start,end):
    direction = end - start
    norm = np.linalg.norm(direction)  # length of direction vector
    if norm != 0:
        direction /= norm  # Normalize the direction vector
    return direction

def perpendicular_direction(start,end):
    direction = vector_direction(start,end)
    return np.array([-direction[1], direction[0]])

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


def create_eyeliner_wing(length, angle, liner_corner,topStart ,bottomStart ,thickness, topCurviness, topCurveStart, bottomCurviness, bottomCurveStart):
    a=0.5
    b=0
    c=0
    eye_corner = np.array([1,0.5])
    liner_corner=np.array(liner_corner)
    angle = np.deg2rad(angle)
    e_x = liner_corner[0] + length * np.cos(angle)
    e_y = liner_corner[1] + length * np.sin(angle)
    P1 = np.array([e_x, e_y])

    if np.array_equal(eye_corner, liner_corner):
        x_lim = find_tangent_point(-a, b, c + 1, e_x, e_y, 0)
        p0x = liner_corner[0] - topStart * (liner_corner[0] - x_lim)
        p0y = (-a) * p0x ** 2 + b * p0x + (c + 1)
        P0=np.array([p0x, p0y])

        x_lim = find_tangent_point(a, b, c, e_x, e_y,2)
        p2x = liner_corner[0] - bottomStart * (liner_corner[0] - x_lim)
        p2y= a * p2x**2 + b * p2x + c
        P2 = np.array([p2x, p2y])
    else:
        perpendicular_vector = perpendicular_direction(liner_corner,P1)
        distance = perpendicular_vector * thickness

        P0= liner_corner + distance
        P0 = P0 + (P0-P1)*topStart
        P2= liner_corner - distance
        P2 = P2 + (P2-P1)*bottomStart

    if topCurviness!=0:
        T0=P0
        T2=P1
        T1=T0 + vector_direction(T0,T2)*((T2-T0)*topCurveStart) #T1 is topCurve start x length between P0 and P1 away from P0
        T1=T1-(perpendicular_direction(T0,T2))*topCurviness #T1 is topCurviness down perpendicular to direction from P0 to P1
        t_values = np.linspace(0, 1, 100)
        topLine = np.array([bezier_curve(t, T0, T1, T2) for t in t_values])
    else:
        topLine = np.array([P0,P1])

    if bottomCurviness!=0:
        B0 = P2
        B2 = P1
        B1 = B0 + vector_direction(B0,B2)*((B2-B0) * bottomCurveStart)
        B1 = B1 - (perpendicular_direction(B0, B2))*bottomCurviness
        t_values = np.linspace(0, 1, 100)
        bottomLine = np.array([bezier_curve(t,B0,B1,B2) for t in t_values])
    else:
        bottomLine = np.array([P1,P2])

    return np.concatenate((topLine,bottomLine))


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
wing_points = create_eyeliner_wing(0.5, 10, (1,1),0.7,0.5,0.2, 0.2,0.5,0.2,0.3)
plt.plot(wing_points[:, 0], wing_points[:, 1], 'b', lw=2)  # Plot all points as a single object
plt.grid(False)
plt.title("Eyeliner Wing")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()
