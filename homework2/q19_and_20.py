from q17_and_18 import *
import numpy as np

"""
    This program solve the high-dimension decision stump problem.
    The answer is shown below.

    Author: SunnerLi
    Finish: 19/10/2016
"""

# Compute the result of specific hypothesis
H = lambda s, x: s if x - theta > 0 else -s

# Variable
trainRowNumber = 100                                # The number of row in training data
testRowNumber = 1000                                # The number of row in testing data
dimNumber = 9                                       # The number of feature in rows

# Array
trainX = np.ndarray([trainRowNumber, dimNumber])    # The training data
trainY = np.ndarray([trainRowNumber])               # The training tag
testX = np.ndarray([testRowNumber, dimNumber])      # The testing data
testY = np.ndarray([testRowNumber])                 # The testing tag

# File name
trainFileName = 'hw2_train.dat'                     # The file name of training data
testFileName = 'hw2_test.dat'                       # The file name of testing data

def read():
    """
        Read the training and testing data
    """
    global trainX
    global trainY
    global testX
    global testY

    # Deal with training data
    count = 0
    with open(trainFileName, 'r') as f:
        while True:
            rawData = f.readline().split(' ')
            rawData = rawData[1:]
            rawData[-1] = rawData[-1][:len(rawData[-1])-1]
            for i in range(dimNumber):
                trainX[count][i] = rawData[i]
            trainY[count] = rawData[-1] 
            count += 1
            if count == trainRowNumber:
                break

    # Deal with testing data
    count = 0
    with open(testFileName, 'r') as f:
        while True:
            rawData = f.readline().split(' ')
            rawData = rawData[1:]
            rawData[-1] = rawData[-1][:len(rawData[-1])-1]
            for i in range(dimNumber):
                testX[count][i] = rawData[i]
            testY[count] = rawData[-1] 
            count += 1
            if count == testRowNumber:
                break

def sort(dimIndex, x, y, _size):
    """
        Sort by x and y would swap as well

        Arg:    The flag of dimension which would be considered,
                data, tags and the number of rows
        Ret:    The ordered data and flags
    """
    for i in range(_size):
        for j in range(i, _size):
            if x[i][dimIndex] > x[j][dimIndex]:
                x[[i, j]] = x[[j, i]]
                y[[i, j]] = y[[j, i]]
    return x, y

def Ein(dimIndex, s):
    """
        Compute the error-in-rate for the specific s 

        Arg:    s described in the function
        Ret:    The value of Ein
    """
    errorTime = 0
    for i in range(size):
        if not H(s, trainX[i][dimIndex]) == trainY[i]:
            errorTime += 1
    return float(errorTime) / size

def train(dimIndex):
    """
        Try to find the minimun Ein of hypothesis with the corresponding dimension index

        Arg:    The dimension index that want to consult
        Ret:    The minimun Ein, with the corresponding s and theta
    """
    global trainX
    global trainY
    minEin = 1.0
    minTheta = 1.0
    minS = 0
    
    trainX, trainY = sort(dimIndex, trainX, trainY, trainRowNumber)
    #print trainX[:][dimIndex]
    minEin, minTheta, minS = find(dimIndex, 1, minEin, minTheta, minS)
    minEin, minTheta, minS = find(dimIndex, -1, minEin, minTheta, minS)
    return minEin, minS, minTheta

def find(dimIndex, s, minEin, minTheta, minS):
    """
        For each probable theta, test the Ein and find the minimun parameter

        Arg:    The s want to test,
                The original minimun Ein and the corresponding theta and s
        Ret:    The final minimun Ein and the corresponding theta and s
    """
    global theta
    for i in range(size):
        if i == 0:
            theta = ( -1 + trainX[ 0][dimIndex] ) / 2
        elif i == size - 1:
            theta = (  1 + trainX[-1][dimIndex] ) / 2
        else:
            theta = ( trainX[i][dimIndex] + trainX[i-1][dimIndex] ) / 2
        if minEin > Ein(dimIndex, s):
            minEin, minTheta, minS = Ein(dimIndex, s), theta, s
    return minEin, minTheta, minS

def test(dimIndex, minTheta, minS):
    """
        Testing the hypothesis with the result

        Arg:    The paramter that we gain at train function
        Ret:    The Eout
    """
    # Initialize the variable
    global testX
    global testY
    global theta
    global s
    theta = minTheta
    s = minS
    testX, testY = sort(dimIndex, testX, testY, testRowNumber)

    # Testing
    errorTime = 0
    for i in range(testRowNumber):
        if not H(s, testX[i][dimIndex]) == testY[i]:
            errorTime += 1
    return float(errorTime) / testRowNumber

if __name__ == "__main__":
    minDim = 10
    minEin = 1
    minS = 2
    minTheta = 1

    # Find the best of best
    read()
    for i in range(dimNumber):
        _Ein, _s, _theta = train(i)
        print "Dimension: ", i, "\tEin: ", _Ein
        if _Ein < minEin:
            minEin, minS, minTheta, minDim = _Ein, _s, _theta, i
    
    # Show the result
    print ""
    print "(Ans 19)\tmin Dimension: ", minDim, '\t\tmin Ein: ', minEin
    minEout = test(minDim, minTheta, minS)
    print "(Ans 20)\tEout: ", minEout