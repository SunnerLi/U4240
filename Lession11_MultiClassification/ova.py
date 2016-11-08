from lr import *
import numpy as np
import logging

"""
    This program use logistic regression to implement one-versus-all algorithm
    It can solve the multi-class classification problem
"""

# Add log configuration
logging.basicConfig(filename='ova.log', level=logging.DEBUG, filemode='w')
mainLogger = logging.getLogger(' main ')
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

def ova():
    """
        Implementation of one-versus-all algorithm
        At each training times, the whole data would be considered as two group
        Thus, the difference of the number toward the two group would be large
    """
    global models
    global mainLogger
    global tags

    models = np.ndarray([groupN, featureN+1])
    for i in range(3):
        logString = "The index of model to training: " + str(i)
        mainLogger.debug(logString)
        print logString

        # Assign a new tag
        _tag = np.copy(tags)
        for j in range(N):
            if tags[j][0] == i:
                _tag[j][0] = 1
            else:
                _tag[j][0] = -1
        _w = lr(x, _tag, lrLogger)
        models[i] = _w
    print "train done"

def classify(x):
    """
        Classify the tag of the specific x row
        The index of the model represent the type of the tag
        It would return a list about the probability of each groups

        Arg:    The feature vector of specific x
        Ret:    The probability result list
    """
    _res = []
    for i in range(groupN):
        _res.append(THETA(np.sum(np.dot(models[i], x))))
    return _res

def validate():
    """
        Validate if the whole training data can be classify as the correct tags
    """
    global mainLogger
    mainLogger.debug(str(models))
    for i in range(N):
        logString = "point index: " + str(i) + '        ' + str(np.array(classify(x[i])).argmax())
        mainLogger.info(logString)
    print "validate done"

if __name__ == "__main__":
    load()
    x = np.asarray(x)
    tags = np.asarray(tags)

    ova()
    validate()