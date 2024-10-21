import numpy as np
import matplotlib.pyplot as plt

# Define the unit set: A triangle with 3 points
triangle = np.array([
    [0, 0],    # Bottom left corner of the triangle
    [0.5, 1],  # Top vertex of the triangle
    [1, 0],    # Bottom right corner of the triangle
    [0, 0]     # Closing the triangle
])

# Function to rotate a set of points by an angle theta
def rotate(points, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return np.dot(points, R)

# Initialize plot
plt.figure(figsize=(6, 6))

# Set the number of star branches (5-point star)
num_branches = 5
angle_between_branches = 2 * np.pi / num_branches  # 72 degrees in radians

# Loop through each branch of the star, applying a rotation
for i in range(num_branches):
    theta = i * angle_between_branches  # Rotation angle for the current branch
    rotated_triangle = rotate(triangle, theta)  # Apply the rotation
    plt.plot(rotated_triangle[:, 0], rotated_triangle[:, 1], 'b')  # Plot the rotated triangle

# Set plot limits and aspect ratio for a symmetric view
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()