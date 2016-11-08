from collections import Counter
from order import *
from lr import *

import matplotlib.pyplot as plt
import plotin1 as pi1
import numpy as np
import logging

"""
    This program use logistic regression to implement one-versus-one algorithm
    It can solve the multi-class classification problem
"""

# Add log configuration
logging.basicConfig(filename='ovo.log', level=logging.DEBUG, filemode='w')
mainLogger = logging.getLogger('main')
lrLogger = logging.getLogger(' lr ')

# Sigmoid function
THETA = lambda x: 1 / (1 + math.exp(-x))

# Training Constant
N = -1                  # The number of row objects
featureN = 2            # The number of feature in a row
groupN = 3              # The number of group we want to classify

# Variable
x = []                  # x
tags = []               # y
models = None           # w
combination = None      # The combinations of the two group data

def load():
    """
        Load the training data
    """
    global N
    global x
    global tags
    count = 0
    with open('train.dat', 'r') as f:
        while True:
            string = f.readline().split(',')

            # Judge if read to the end
            if string == ['']:
                N = count
                break

            # Assign the data
            _x = []
            _x.append(1)
            _x.append(float(string[0]))
            _x.append(float(string[1]))
            _tag = []
            _tag.append(float(string[2]))
            x.append(_x)
            tags.append(_tag)
            count += 1

def ovo():
    """
        Implementation of one-versus-one algorithm
        At each training times, the specific two groups of data would be selected
        As the result, the difference of the number toward the two group would be small
    """
    global models
    global mainLogger
    global tags
    global combination

    # Get all combination of group
    combination = getOrder(groupN, 2)
    models = np.ndarray([len(combination), featureN+1])

    # Train
    for i in range(len(combination)):
        logString = "The index of model to training: " + str(i)
        mainLogger.debug(logString)
        print logString

        # Assign a new tag
        _tag = []
        _x = []
        for j in range(N):
            if tags[j][0] == combination[i][0]:
                _tag.append([1])
                _x.append(x[j])
            elif tags[j][0] == combination[i][1]:
                _tag.append([-1])
                _x.append(x[j])

        # Transfer as the origin type (numpy object)
        _tag = np.array(_tag)
        _x = np.array(_x)

        # Logistic regression
        _w = lr(_x, _tag, lrLogger)
        models[i] = _w
    print "train done"

def classify(x):
    """
        Classify the tag of the specific x row
        The index of the model represent if the data is in the front group or not
        It would return the index of the group which the point is probably in

        Arg:    The feature vector of specific x
        Ret:    The probability result list
    """
    # Get each judgement of classifier
    _res = []
    for i in range(groupN):
        _res.append(THETA(np.sum(np.dot(models[i], x))))

    # Re-tagged the result of modul vector
    _res1 = []
    for i in xrange(len(combination)):
        if _res[i] > 0.5:
            _res1.append(combination[i][0])
        else:
            _res1.append(combination[i][1])
    
    # Maximun likelihood estimation
    res = Counter(_res1).most_common(1)[0][0]        
    return res

def validate():
    """
        Validate if the whole training data can be classify as the correct tags
    """
    global mainLogger
    mainLogger.debug(str(models))
    for i in range(N):
        logString = "point index: " + str(i) + '\t' + str(classify(x[i]))
        mainLogger.info(logString)
    print "validate done"

if __name__ == "__main__":
    load()
    x = np.asarray(x)
    tags = np.asarray(tags)

    ovo()
    validate()