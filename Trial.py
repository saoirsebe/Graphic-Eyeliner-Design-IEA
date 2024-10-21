import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image using OpenCV
#image_path = "ee77d893c56513c4422d87235e6eeeee.jpg"  # Path to your image file
#img = cv2.imread(image_path)

# Convert the image from BGR (OpenCV format) to RGB (Matplotlib format)
#img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image and manually select points
#plt.imshow(img_rgb)

# Function to calculate a point on a cubic Bezier curve
def bezier_curve(t, P0, P1, P2, P3):
    return (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3

# Control points for the eyeliner curve
# Adjust these points to match the eyeliner curve and wing
P0 = np.array([0, 0])      # Start point (inner corner of the eye)
P1 = np.array([0.5, 0.1])  # First control point (along the eyelid curve)
P2 = np.array([1, -0.1])  # Second control point (before the wing)
P3 = np.array([2.0, 0.1])    # End point (sharp wing tip)

# Generate t values (from 0 to 1)
t_values = np.linspace(0, 1, 100)

# Calculate the Bezier curve points
curve_points = np.array([bezier_curve(t, P0, P1, P2, P3) for t in t_values])

# Plot the Bezier curve
plt.plot(curve_points[:, 0], curve_points[:, 1], color='black', linewidth=2)

# Plot control points
plt.scatter([P0[0], P1[0], P2[0], P3[0]], [P0[1], P1[1], P2[1], P3[1]], color='red')

# Annotate the control points
plt.text(P0[0], P0[1], 'P0 (Start)', fontsize=12, verticalalignment='bottom')
plt.text(P1[0], P1[1], 'P1 (Control)', fontsize=12, verticalalignment='bottom')
plt.text(P2[0], P2[1], 'P2 (Control)', fontsize=12, verticalalignment='bottom')
plt.text(P3[0], P3[1], 'P3 (End)', fontsize=12, verticalalignment='bottom')

# Set axis limits and labels
plt.xlim(-0.1, 3.1)
plt.ylim(-0.1, 0.6)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cubic Bezier Curve for Eyeliner Shape')

# Show the plot
plt.grid(True)
plt.show()

