from collections import Counter
import matplotlib.pyplot as plt
import plotin1 as pi1
from order import *
import numpy as np
from lr import *
import logging
logging.basicConfig(filename='ovo.log', level=logging.DEBUG, filemode='w')
mainLogger = logging.getLogger('main')

THETA = lambda x: 1 / (1 + math.exp(-x))


N = 30
featureN = 2
groupN = 3
x = np.ndarray([N, featureN+1])
tags = np.ndarray([N, 1])
models = None
combination = None

def load():
    global x
    global tags

    count = 0
    with open('train.dat', 'r') as f:
        while True:
            string = f.readline().split(',')
            x[count][0] = 1
            x[count][1] = string[0]
            x[count][2] = string[1]
            tags[count][0] = string[2]
            count += 1
            if count >= N:
                break

def ovo():
    global models
    global mainLogger
    global tags
    global combination

    # Get all combination of group
    combination = getOrder(groupN, 2)
    print len(combination)

    models = np.ndarray([len(combination), featureN+1])
    for i in range(len(combination)):
        mainLogger.debug("tern: " + str(i))

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
        _tag = np.array(_tag)
        _x = np.array(_x)
        print np.transpose(_tag)

        #pi1.PointChart(_x[:, 1].tolist(), _x[:, 2].tolist(), _tag[:, 0].tolist()).show()
        _w = lr(_x, _tag)
        models[i] = _w
    print "train done"

def classify(x):
    _res = []
    
    _res.append(THETA(np.sum(np.dot(models[0], x))))
    _res.append(THETA(np.sum(np.dot(models[1], x))))
    _res.append(THETA(np.sum(np.dot(models[2], x))))
    """
    print "res: ", _res
    print type(_res)
    print "res: ", _res
    """
    #_res1 = _res[:]
    _res1 = []
    for i in xrange(len(combination)):
        #print "res: ", _res
        if _res[i] > 0.5:
            #_res1 = combination[i][0]
            _res1.append(combination[i][0])
        else:
            #_res1 = combination[i][1]
            _res1.append(combination[i][1])
        #print _res1
        #np.array(_res).argmax()
    #print "freq: ", Counter(_res1).most_common(1)
    res = Counter(_res1).most_common(1)[0][0]
    #print "res: ", res
        

    return res

def validate():
    print models
    for i in range(N):
        print "point index: ", i, '\t', classify(x[i])
        #print classify(x[i])

load()
ovo()
validate()