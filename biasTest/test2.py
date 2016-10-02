import pylab
import numpy as np
import math

def sigmoid(x):
    """
        Return the sigmoid function for the input array

        Arg:    The input numpy array
        Ret:    The sigmoid result
    """
    return 1 / ( 1 + np.exp(-x) ) 

def sigmoid_with_bias(x, b):
    """
        Return the sigmoid function with the bias for the input array

        Arg:    The input numpy array and the bias vector
        Ret:    The sigmoid result
    """
    x = x + b
    return 1 / ( 1 + np.exp(-x) ) 

# Plot the function
x = np.linspace(-15, 15, 100)
print x.shape
y1 = sigmoid(x)
y2 = sigmoid_with_bias(x, 2 * np.ones(100))
y3 = sigmoid_with_bias(x, 2 * -(np.ones(100)))

# Show the value when x = 0
print "bias =  0,  x = 0, \ty1 = ", sigmoid(np.zeros(1))
pylab.plot(x, y1)

print "bias =  2,  x = 0, \ty2 = ", sigmoid_with_bias(np.zeros(1), 2 * np.ones(1))
pylab.plot(x, y2)

print "bias = -2,  x = 0, \ty3 = ", sigmoid_with_bias(np.zeros(1), 2 * -np.ones(1))
pylab.plot(x, y3)
pylab.show()