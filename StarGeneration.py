import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import cosine
from A import *
from EyelinerWingGeneration import normalised_vector_direction


# Function to create a quadratic Bézier curve with three control points
def bezier_curve(t, P0, P1, P2):
    x = (1 - t) ** 2 * P0[0] + 2 * (1 - t) * t * P1[0] + t ** 2 * P2[0]
    y = (1 - t) ** 2 * P0[1] + 2 * (1 - t) * t * P1[1] + t ** 2 * P2[1]

    return [round(x, 3), round(y, 3)]

def create_star_arm(center, radius, arm_length, num_points, start_angle, asymmetry, arm_n,star_type):
    centerx = center[0]
    centery = center[1]
    angle = 2 * np.pi / num_points

    if star_type == StarType.CURVED or star_type == StarType.FLOWER:
        print("We curved")
        starSize = arm_length
        if radius>0:
            starSize = starSize + radius

        P0 = np.array([centerx + starSize * np.cos(start_angle), centery + starSize * np.sin(start_angle)])  # Inner tip of the arm
        P2 = np.array([centerx + starSize * np.cos(start_angle + angle),centery + starSize * np.sin(start_angle + angle)])  # Outer tip of the arm (on the positive x-axis)
    else:
        P0 = np.array([centerx + radius*np.cos(start_angle), centery + radius*np.sin(start_angle) ])  # Inner tip of the arm
        P2 = np.array([centerx + radius*np.cos(start_angle+angle), centery + radius*np.sin(start_angle+angle)])  # Outer tip of the arm (on the positive x-axis)
        print(P0,P2)
    M = (P0 + P2) / 2
    direction = normalised_vector_direction(center,M)

    if star_type == StarType.STRAIGHT:
        P1 = M + direction * arm_length
    elif star_type == StarType.CURVED:
        P1= center + direction * radius
        #P1 = M - direction * arm_length
        #plt.plot(P1)

    if (asymmetry!=0) and (np.mod(arm_n,2)==0) and (np.mod(num_points,2)==0) and (star_type == StarType.STRAIGHT):
        print("1")
        P1 = P1 + (direction * asymmetry)
    elif (asymmetry!=0) and (np.mod(arm_n,2)==0) and (np.mod(num_points,2)==0) and (star_type == StarType.CURVED):
        p_direction = normalised_vector_direction(center,P0)
        P0 = P0 + (p_direction * asymmetry)
        print("2")
    elif (asymmetry!=0) and (np.mod(arm_n,2)!=0) and (np.mod(num_points,2)==0) and (star_type == StarType.CURVED):
        p_direction = normalised_vector_direction(center,P2)
        P2 = P2 + (p_direction * asymmetry)
        print("3")

    if star_type == StarType.CURVED:
        t_values = np.linspace(0, 1, 100)
        arm_points = np.array([bezier_curve(t, P0, P1, P2) for t in t_values])
    else:
        arm_points = np.array([P0, P1, P2])

    if asymmetry==0 and star_type == StarType.CURVED:
        x, y = P2
    elif star_type == StarType.STRAIGHT:
        x, y = P1
    else:
        x, y = P0
    #print(arm_points)
    return arm_points, (x,y)

def create_star(num_points, center, radius, arm_length , asymmetry, star_type, star_direction):
    star_points = []
    end_point = (0,0) #Starts at (0,0) and changed to P2 after each arm creation
    start_point = (0,0)
    star_arm_points =[]
    # Generate and plot each arm of the star using Bézier curves
    for i in range(num_points):
        armN=i
        angle = (2 * np.pi * i / num_points) + math.radians(star_direction)
        """
        if i == end_arm:
            arm_points , end_point = create_star_arm(center, radius,arm_length, num_points, angle, asymmetry, armN, curved)
        elif i == num_points-1:
            arm_points, start_point = create_star_arm(center, radius, arm_length, num_points, angle, asymmetry, armN, curved)
        else:
            arm_points, bin_point = create_star_arm(center, radius, arm_length, num_points, angle, asymmetry, armN, curved)
        """
        arm_points, arm_point = create_star_arm(center, radius, arm_length, num_points, angle, asymmetry, armN, star_type)
        star_points.extend(arm_points)
        star_arm_points.append(arm_point)

    star_points = np.array(star_points) # Convert star_points to a numpy array for plotting
    return star_points, star_arm_points #end_point , start_point


"""
# Set up the plot
plt.figure(figsize=(6, 6))
ax = plt.gca()
ax.set_aspect('equal')
# Plot a star with 5 arms, curved edges using quadratic Bézier curves
star_points_plot,p2 = create_star_arm((0,0), 0,2,4,math.radians(45),0,0,True)
#star_points_plot ,p2= create_star(4, (0, 0), 0,0.5, 0,True)
plt.plot(star_points_plot[:, 0], star_points_plot[:, 1], 'b', lw=2) # Plot all points as a single object
plt.grid(False)
plt.show()
plt.plot(p2,color='red')
print(p2)
"""

