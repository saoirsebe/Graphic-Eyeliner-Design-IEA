import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import cosine


# Function to create a quadratic Bézier curve with three control points
def bezier_curve(t, P0, P1, P2):
    return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2


# Function to create one arm of the star using a quadratic Bézier curve
def create_star_arm(center, radius, curve_control, num_points, startAngle,num_curve_points,asymmetry,isArmAsymetric):
    centerx = center[0]
    centery = center[1]
    angle = 2 * np.pi / num_points

    P0 = np.array([centerx + radius*np.cos(startAngle)  , centery + radius*np.sin(startAngle) ])  # Inner tip of the arm
    P2 = np.array([centerx + radius*np.cos(startAngle+angle) ,centery + radius*np.sin(startAngle+angle)])  # Outer tip of the arm (on the positive x-axis)

    control_point_distance = curve_control * radius  # Controls the curvature

    M = (P0 + P2) / 2
    direction = M - center
    norm = np.linalg.norm(direction) #length of direction vector
    if norm != 0:
        direction /= norm  # Normalize the direction vector
    P1 = M - direction * control_point_distance

    if (asymmetry!=0) and (np.mod(isArmAsymetric,2)==0) and (np.mod(num_points,2)==0):
        pDirection = P0-center
        P0 = P0 + (pDirection * asymmetry)
    elif (asymmetry!=0) and (np.mod(isArmAsymetric,2)!=0) and (np.mod(num_points,2)==0):
        pDirection = P2 - center
        P2 = P2 + (pDirection * asymmetry)

        # Generate the Bézier curve points for the arm
    t_values = np.linspace(0, 1, num_curve_points)
    arm_points = np.array([bezier_curve(t, P0, P1, P2) for t in t_values])
    plt.scatter(center[0], center[1], color='red', label='Center')
    return arm_points

def rotation_matrix(theta):
    """ Return the 2D rotation matrix for an angle theta in degrees """
    theta_rad = np.deg2rad(theta)
    return np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])

def rotate_points(points, theta):
    """ Rotate a set of 2D points by theta degrees """
    rot_matrix = rotation_matrix(theta)
    return np.dot(points, rot_matrix.T)

# Function to plot the star with curved edges using quadratic Bézier curves
def create_curved_star(num_points, center, radius, curve_control, num_curve_points,isAsymetric):
    # Set up the plot
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    star_points = []
    # Generate and plot each arm of the star using Bézier curves
    for i in range(num_points):
        isArmAsymetric=i
        angle = 2 * np.pi * i / num_points
        arm_points = create_star_arm(center, radius, curve_control,num_points,angle,num_curve_points,isAsymetric,isArmAsymetric)
        star_points.extend(arm_points)

    star_points = np.array(star_points) # Convert star_points to a numpy array for plotting
    return star_points


# Plot a star with 5 arms, curved edges using quadratic Bézier curves
star_points = create_curved_star(7, (0, 0), 1, 0.7, 100,0.5)
plt.plot(star_points[:, 0], star_points[:, 1], 'b', lw=2) # Plot all points as a single object
plt.grid(False)
plt.show()
"""
plt.figure(figsize=(6, 6))
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
arm_points =create_star_arm((0,0),1,0.5,6,0,100)
# Plot the star arm
plt.plot(arm_points[:, 0], arm_points[:, 1], '-o')

plt.show()
"""