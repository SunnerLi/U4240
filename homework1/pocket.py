from matplotlib.pyplot as plt
from homework15_IO import *
from homework16 import *
import numpy as np

sign = lambda x: 1 if x > 0 else -1

# Variable
threshold = 1

# Constant
Use_Training_Dataset_To_Validate = 100
Use_Testing_Dataset_To_Validate = 101
Better = 102
Worst = 103

def validate(X, Y, W, X_, Y_, W_):
    """
        Validate if the new weight vector is the better choise

        Arg:    D{ (Xi, yi) } , the old weight,
                The validate dataset input and the new weight
        Ret:    If it's better
    """
    # Calculate the error toward the old weight
    old = 0
    for i in range(len(X)):
        if not sign(np.dot(np.array(X[i]), W)) == Y[i]:
            old += 1 
    
    # Calculate the error toward the new weight
    new = 0
    for i in range(len(X_)):
        if not sign(np.dot(np.array(X_[i]), W_)) == Y_[i]:
            new += 1 
    
    # Return the result
    if old < new:
        return Worst
    return Better

def pocketDraw(title, array):
    """
        Draw the histogram toward the pocket algorithm result

        Arg:    The 
    """

def pocket(X, Y, XTest, YTest, eta=1, updateTime=100, skipValid=False,
        validate_type=Use_Testing_Dataset_To_Validate):
    """
        Implement the pocket algorithm

        Arg:    The input data, the tag, the learning rate,
                The update round time and the type of the dataset to validate
        Ret:    The error rate
    """
    W = np.zeros(5)
    W[0] = threshold
    WPocket = W
    updateCount = 0

    # Pocket algorithm
    for i in range(len(X)):
        Xi = np.array(X[i])
        if not sign(np.dot(Xi, W)) == Y[i]:
            W = W + eta * Xi * Y[i]
            if not skipValid:
                if validate(X, Y, WPocket, XTest, YTest, W) == Better:
                    WPocket = W
            else:
                WPocket = W

            # If go up to the update limit
            updateCount += 1
            if not updateCount >= updateTime:
                i = 0

    # Validate
    minCount = 0
    if validate_type == Use_Testing_Dataset_To_Validate:
        for i in range(len(XTest)):
            if not sign(np.dot(np.array(XTest[i]), WPocket)) == YTest[i]:
                minCount += 1
    else:
        for i in range(len(X)):
            if not sign(np.dot(np.array(X[i]), WPocket)) == Y[i]:
                minCount += 1
    return float(minCount) / 500

def pocket2(X, Y, XTest, YTest, eta=1, updateTime=50, skipValid=False,
        validate_type=Use_Testing_Dataset_To_Validate):
    """
        Implement the pocket algorithm

        Arg:    The input data, the tag, the learning rate,
                The update round time and the type of the dataset to validate
        Ret:    The error rate
    """
    W = np.zeros(5)
    W[0] = threshold
    WPocket = W
    updateCount = 0

    # Pocket algorithm
    """
    for i in range(len(X)):
        Xi = np.array(X[i])
        if not sign(np.dot(Xi, W)) == Y[i]:
            W = W + eta * Xi * Y[i]
            if not skipValid:
                if validate(X, Y, WPocket, XTest, YTest, W) == Better:
                    WPocket = W
            else:
                WPocket = W

            # If go up to the update limit
            updateCount += 1
            if not updateCount >= updateTime:
                i = 0
    """
    index = 0
    while updateCount < updateTime:
        Xi = np.array(X[index])
        if not sign(np.dot(Xi, W)) == Y[index]:
            updateCount += 1
            W = W + eta * Xi * Y[index]
            if not skipValid:
                if validate(X, Y, WPocket, XTest, YTest, W) == Better:
                    WPocket = W
            else:
                WPocket = W
            updateCount += 1
        index = ( index+1 ) % len(X)


    # Validate
    minCount = 0
    if validate_type == Use_Testing_Dataset_To_Validate:
        for i in range(len(XTest)):
            if not sign(np.dot(np.array(XTest[i]), WPocket)) == YTest[i]:
                minCount += 1
    else:
        for i in range(len(X)):
            if not sign(np.dot(np.array(X[i]), WPocket)) == Y[i]:
                minCount += 1
    return float(minCount) / 500

"""
if __name__ == "__main__":
    # Variable
    errors = []
    WPocket = None

    D, X, Y = read(fileName="18_train.dat")
    DTest, XTest, YTest = read(fileName="18_test.dat")
    for i in range(200):
        X, Y = shuffle(X, Y)
        error = pocket2(X, Y, XTest, YTest)
        print "Epoch: ", i, "\tError: ", error
        errors.append(error)
    print "Error Rate: ", float( np.sum(np.array(errors)) / 200 ) 
    draw("Error Rate", errors)
"""