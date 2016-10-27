import numpy as np
import math

"""
    This code solve the problem 18 and 19
    But the answer is a little far from the correct answer of others
    So It should be inspected carefully. 
"""

# The sigmoid fundtion toward the vector
THETA = lambda x: 1 / (1 + np.exp(-x))      

# Training constants
iteration = 2000
eta = 0.001

# Training data variable
N = 1000
x = np.ndarray([N, 20])
y = np.ndarray([N, 1])

# Testing data variable
testN = 3000
testX = np.ndarray([testN, 20])
testY = np.ndarray([testN, 1])

# The time period to show the loss information
showLossLimit = 500

def read():
    """
        Read the training data and testing data from file
    """
    # Read training data
    with open('hw3_train.dat', 'r') as f:
        for i in range(N):
            string = f.readline().split()
            x[i] = np.asarray(string[:-1])
            y[i] = string[-1]

    # Read testing data
    with open('hw3_test.dat', 'r') as f:
        for i in range(testN):
            string = f.readline().split()
            testX[i] = np.asarray(string[:-1])
            testY[i] = string[-1]

def gradient():
    """
        Calculate the gradient for the cross entropy

        Ret:    The gradient vector that can be used to revise the model
    """
    thetaIn = np.multiply(np.transpose(y), np.matmul(w, np.transpose(x)))
    thetaOut = THETA(-thetaIn)
    backIn = np.multiply(y, x)
    sumIn = np.multiply(np.transpose(thetaOut), -backIn)
    return np.asarray([np.sum(sumIn, axis=0)])

def validate():
    """
        Validate by the training data

        Ret:    The data-in error (Ein)
    """
    y_ = np.sign(np.matmul(x, np.transpose(w)))
    count = 0.0
    for i in range(N):
        if not y[i][0] == y_[i][0]:
            count += 1
    return float(count) / N

def test():
    """
        Testing by the testing data

        Ret:    The data-out error (Eout)
    """
    y_ = np.sign(np.matmul(testX, np.transpose(w)))
    count = 0.0
    for i in range(testN):
        if not testY[i][0] == y_[i][0]:
            count += 1
    return float(count) / testN

def train():
    """
        Implement model training
    """
    global w
    for i in range(iteration):
        w = w - eta * gradient()
        if i % showLossLimit == 0:
            print "Iter: ", i, "loss: ", np.square(np.sum(gradient()))
    print ""

if __name__ == "__main__":
    read()
    
    print "----- Q18 -----"
    w = np.random.rand(1, 20)
    train()
    print "Eout: ", test()
    
    print "----- Q19 -----"
    eta = 0.01
    iteration = 20000
    showLossLimit = 5000
    train()
    print "Eout: ", test()