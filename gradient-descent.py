# Chandler Severson
# Gradient Descent - Machine Learning
# Jul 20, 2017

from numpy import *

#
#   Notes:
#   -A DERIVATIVE is the SLOPE of a function at a GIVEN POINT
#       -PARTIAL DERIVATIVE is the slope in respect to one of
#       the variables in the line
#
#   -GRADIENTS are Derivatives that point towards the direction
#       of a local minima of a function
#
#   -GRADIENT DESCENT is a popular optimization strategy in ML
#       that uses GRADIENTS to find the local minima of a function.
#
# Gradient Descent GIF: http://bit.ly/2uOO1gL


# Compute The Error for a Line, given a set of Data Points.
# "Sum of Squared Distances" Formula (To Calculate out Error)
#   *Error Measure for how close a predicted line is to the actual data
#   *https://spin.atomicobject.com/wp-content/uploads/linear_regression_error1.png
#
# GOAL: Find optimal 'b' and 'm' for a line, such that the line hits as many points as possible
# SLOPE formula: y=mx+b
#
def compute_err_for_line(b,m,pts):
    error = 0
    for i in range(0, len(pts)): #for every single data point
        x = pts[i,0]
        y = pts[i,1] #get the assoicated Y value from the point set

        # subtract the actual Y value (from slope formula) from the
        # Y value found above, to measure the distance (or error)
        # between the two points
        error += (y - ( m * x + b)) ** 2

    # TheError(m,b) = 1/n * error, where 'n' is the total number of data points
    return error / float(len(pts))


# Take the Partial Derivative with Respect to B and M,
# Performing Gradient Descent.
#   *https://spin.atomicobject.com/wp-content/uploads/linear_regression_gradient1.png
#
def step_gradient(b_curr, m_curr, pts):
    b_gradient = 0
    m_gradient = 0
    N = float(len(pts))
    for i in range(0, len(pts)):
        x = pts[i,0]
        y = pts[i,1]

        # Calculate Derivative with respect to M. (Power Rule)
        m_gradient = -(2/N) *  x*(y - (m_curr * x + b_curr))

        # Calculate Derivative with respect to B. (Power Rule)
        b_gradient = -(2/N) *  (y - (m_curr * x + b_curr))

    # Update both parameters to make B and M descend towards an optimized line
    new_b = b_curr - b_gradient
    new_m = m_curr - m_gradient
    return [new_b, new_m]

# Do Gradient Descent (step_gradient) for a 'num_iterations' number of times
# Performing this will minimize the function towards the local minima
# where the distance/error between our function and all of the data points is the smallest
#
def perform_descent(points, b_start, m_start, learning_rate, num_iterations):
    b = b_start
    m = m_start

    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points))

    return b,m


def run():
    # Example Data: Video Game Sales. Critic Score vs Global Sales
    #https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings
    points = genfromtxt("data2.csv", delimiter=",", skip_header=1)

    learn_rate = 0.0001
    b_initial = 0
    m_initial = 0
    num_iterations = 8000
    print('Started gradient descent. b = {0}, m = {1}, error = {2}'.format(b_initial, m_initial, compute_err_for_line(b_initial, m_initial, points)))
    print("Running...")
    [b, m] = perform_descent(points, b_initial, m_initial, learn_rate, num_iterations)
    print("Finished after {0} iterations. b = {1}, m = {2}, error = {3}".format (num_iterations, b, m, compute_err_for_line(b,m,points)))

if __name__ == '__main__':
    run()
