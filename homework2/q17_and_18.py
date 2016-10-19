from datetime import datetime
import numpy as np
import random

"""
    This program solve the 1-dimension decision stump problem.
    The answer is shown below.

    Author: SunnerLi
    Finish: 19/10/2016
"""
x = None            # The uniform-distribution dataset
y = None            # The corresponding tag
theta = None        # The theta variable in hypothesis
size = 20           # The size of dataset
testTimes = 5000    # The repeat times

# Compute the result of specific hypothesis
H = lambda s, x: s if x - theta > 0 else -s

# Compute the Eout by the equation that we get in Question 16
Eout = lambda s, theta: 0.5 + 0.3 * s * (abs(theta) - 1)

def generate():
    """
        Generate the dataset and tag with 20% probable flipping
    """
    global x
    global y
    random.seed(datetime)
    x = np.sort( np.random.uniform(-1, 1, size) )
    y = np.sign(x)
    for i in xrange(len(x)):
        needToFlip = random.randint(0, 4)
        if needToFlip == 1:
            y[i] = -y[i]

def Ein(s):
    """
        Compute the error-in-rate for the specific s 

        Arg:    s described in the function
        Ret:    The value of Ein
    """
    errorTime = 0
    for i in range(size):
        if not H(s, x[i]) == y[i]:
            errorTime += 1
    return float(errorTime) / size

def testing():
    """
        The interface of the whole work

        Ret:    The minimun Ein and the corresponding Eout
    """
    minEin = 1.0
    minTheta = 1.0
    minS = 0
    
    minEin, minTheta, minS = find(1, minEin, minTheta, minS)
    minEin, minTheta, minS = find(-1, minEin, minTheta, minS)
    return minEin, Eout(minS, minTheta)

def find(s, minEin, minTheta, minS):
    """
        For each probable theta, test the Ein and find the minimun parameter

        Arg:    The s want to test,
                The original minimun Ein and the corresponding theta and s
        Ret:    The final minimun Ein and the corresponding theta and s
    """
    global theta
    for i in range(size):
        if i == 0:
            theta = (-1+x[0])/2
        elif i == size - 1:
            theta = (1+x[-1])/2
        else:
            theta = (x[i]+x[i-1])/2
        if minEin > Ein(s):
            minEin, minTheta, minS = Ein(s), theta, s
    return minEin, minTheta, minS

if __name__ == "__main__":
    sumEin = 0
    sumEout = 0
    for i in xrange(testTimes):
        generate()
        _Ein, _Eout = testing()
        sumEin += _Ein
        sumEout += _Eout
    print "(Ans17) Average Ein: ", sumEin / testTimes
    print "(Ans18) Average Eout: ", sumEout / testTimes