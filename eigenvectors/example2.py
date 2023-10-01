# INTRO
# in the previous examples we saw how a matrix
# m can transform an input vector x
# into an output vector y via matrix multiplication
# y = m x
# We did it with a few vectors. Here we show
# the same operation but with 2000 input vector
# the input is a cloud of 2000 points (the tip of the vectors)
# and the output is another cloud of 2000 Points
# We also plot the eigenvectors of the matrix m
# Question: How do the eigenvectors relate to the plot of transformed points
# Extra Points: print the eigenvalues for each eigenvector. Hypothesize
# how they relate to the plot of transformed points


import numpy as np
import matplotlib.pyplot as plt

# Generate a cloud of points uniformly from a circle of radius 1
num_points = 1000
theta = np.random.uniform(0, 2*np.pi, num_points)
r = np.sqrt(np.random.uniform(0, 1, num_points))
x = r * np.cos(theta)
y = r * np.sin(theta)

# Original cloud of points
original_points = np.column_stack((x, y))

# Define the matrix m (same as A)
m = np.array([[2, 1],
              [1, 3]])

# Transform the cloud of points
transformed_points = np.dot(original_points, m.T)

# Compute eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(m)

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot original cloud of points
axs[0].scatter(original_points[:, 0], original_points[:, 1], c='r', alpha=0.5, s=5)
axs[0].set_title('Original Cloud of Points')
axs[0].set_xlim(-5, 5)
axs[0].set_ylim(-5, 5)
axs[0].grid(True)

# Plot transformed cloud of points
axs[1].scatter(transformed_points[:, 0], transformed_points[:, 1], c='b', alpha=0.5, s=5)
axs[1].set_title('Transformed Cloud of Points')
axs[1].set_xlim(-5, 5)
axs[1].set_ylim(-5, 5)
axs[1].grid(True)

# Add arrows for eigenvectors with increased linewidth
for eigvec in eigenvectors.T:
    axs[1].arrow(0, 0, eigvec[0]*5, eigvec[1]*5, head_width=0.5, head_length=0.5, fc='g', ec='g', linewidth=2)

# Show the plot
plt.show()
