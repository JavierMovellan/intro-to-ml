# INTRO
# The goal of this script is to understand eigenvectors and eigvenvalues of
# matrices
# When we multiply a matrix times a vector we get another vector
# For example if the input vector x  is x= [1,1]''
# and the matrix m is m = [ 2, 1]
#                         [ 1, 3]

# then the output vector y is
# y = m x = [ 3, 4]
# So for each vector x we get a different vector y
# and the matrix m is the one that transforms inputs x into outputs y
# In this script x and y are just 2dimensional points.
# In the future x could be an input image
# and y an output image. Or x could be a text prompt and y an image produced
# in response to the propmpt.

# In general a matrix m will change the length and orientation of the input
# so the input vector x and output vector y are not aligned
# however each matrix m has a special type of input vectors so that the outputs
# and the inputs are aligned. These are called the eigenvectors of m
# So if the input is an eigenvector x then the output y  will be a streched or
# shrunked version of the input, which will also be an eigenvector.

# So if the input is an eigenvector the output will also be an eigenvector
# we will not be able to modify the direction of this input no matter how
# many times we apply the transformation matrix m to it.

# This turns out to be an important property in some neural network models

###### Script
# use the python numpy library and rename it "np" in this script
import numpy as np
# yse the matplotlib.pyplot library and rename it plt in this script
import matplotlib.pyplot as plt

# Function to plot vectors
def plot_vectors(vectors, colors, title):
    plt.figure()
    zeros_x = np.zeros(len(vectors))
    zeros_y = np.zeros(len(vectors))
    for i in range(len(vectors)):
        plt.quiver(zeros_x[i], zeros_y[i], vectors[i][0], vectors[i][1], angles='xy', scale_units='xy', scale=1, color=colors[i])
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.axvline(x=0, color='grey', lw=1)
    plt.axhline(y=0, color='grey', lw=1)
    plt.grid(True)
    plt.title(title)
    plt.show()
    plt.waitforbuttonpress()
    plt.close()

# Define a 2x2 matrix m
m = np.array([[2, 1],
              [1, 3]])

# Define some 2D vectors to be transformed
# each row is an input vector x
# and for each we will get an output vector y
# which is y  = m x the matrix product between
# the matrix m and the vector x
vectors = np.array([[1, 0],
                    [0, 1],
                    [1, 1],
                    [1, -1],
                    [-1, -1]])

# Loop through each vector to show its transformation
for i, vec in enumerate(vectors):
    print(f"Showing vector {i+1}")

    # Plot original vector
    plot_vectors([vec], ['r'], "Original Vector")

    # Transform the input vector
    # dot is the matrix multiplication Function
    # in the np library
    transformed_vec = np.dot(m, vec)

    # Plot both original and transformed vector
    plot_vectors([vec, transformed_vec], ['r', 'b'], "Original and Transformed Vectors")

# Compute eigenvectors and eigenvalues
# for this we use the eig function of the linalg package
# of the np (numpy) library
eigenvalues, eigenvectors = np.linalg.eig(m)

# Loop through each eigenvector to show its transformation
for i, eigvec in enumerate(eigenvectors.T):
    print(f"Showing eigenvector {i+1}")

    # Plot original eigenvector
    plot_vectors([eigvec], ['g'], "Original Eigenvector")

    # Transform the eigenvector
    transformed_eigvec = np.dot(m, eigvec)

    # Plot both original and transformed eigenvector
    plot_vectors([eigvec, transformed_eigvec], ['g', 'm'], "Original and Transformed Eigenvector")

print("Notice how the transformed eigenvectors (in magenta) align with the original eigenvectors (in green). This shows that they only get 'stretched' or 'compressed' but do not change direction.")
