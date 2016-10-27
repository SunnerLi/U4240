import matplotlib.pyplot as plt
import numpy as np
import random

# Define the error that the length of the two nd-array isn't the same
class ArrayLengthNotTheSameError(object):
    pass

# The quadratic circle function
f = lambda x1, x2: 1 if pow(x1, 2) + pow(x2, 2) - 0.6 >= 0 else -1

# The constants
N = 1000        # The number of points
times = 1000    # The times to repeat

def draw():
    """
        Draw the point of the examine
    """
    for i in range(N):
        if Y[i][0] == 1:
            plt.plot(X[i][1], X[i][2], 'or', color='b')
        else:
            plt.plot(X[i][1], X[i][2], 'or', color='r')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.show()

def linearRegression():
    """
        Linear regression implementation
    """
    global W
    tagger = np.linalg.pinv( np.matmul(np.transpose(X), X) )
    pseudoInverse = np.matmul( tagger, np.transpose(X) )
    W = np.matmul(pseudoInverse, Y)

def predict():
    """
        Predict the result by the linear regression model

        Ret:    The predict list
    """
    _Y = np.sign(np.matmul(X, W))
    return _Y

def Ein(y1, y2):
    """
        Compute the Ein by the predict list and the tag list

        Arg:    The two input list
        Ret:    The rate of Ein
    """
    count = 0
    if not np.shape(y1) == np.shape(y2):
        raise ArrayLengthNotTheSameError
    for i in range(len(y1)):
        if not y1[i][0] == y2[i][0]:
            count += 1
    return float(count) / len(y1)

if __name__ == "__main__":
    EinAvg = 0.0
    for i in xrange(times):
        # Generate points
        X = (np.random.rand(N, 3) - 0.5) * 2
        Y = np.ndarray([N, 1])
        X[0][:] = 1
        W = None
    
        # Generate tags
        for i in range(N):
            Y[i][0] = f(X[i][1], X[i][2])
            if random.randint(0, 9) == 0:
                Y[i][0] = -Y[i][0]
            
        # Regression & predict
        linearRegression()
        _Y = predict()
        EinAvg += Ein(_Y, Y)
    
    # Show the result
    print "Ein average: ", EinAvg / times
    draw()