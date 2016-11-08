import numpy as np
from lr import *
import logging
logging.basicConfig(filename='ova.log', level=logging.DEBUG, filemode='w')
mainLogger = logging.getLogger('main')

THETA = lambda x: 1 / (1 + math.exp(-x))


N = 30
featureN = 2
groupN = 3
x = np.ndarray([N, featureN+1])
tags = np.ndarray([N, 1])
models = None

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

def ova():
    global models
    global mainLogger
    global tags

    models = np.ndarray([groupN, featureN+1])
    for i in range(3):
        mainLogger.debug("tern: " + str(i))

        # Assign a new tag
        _tag = np.copy(tags)
        for j in range(N):
            if tags[j][0] == i:
                _tag[j][0] = 1
            else:
                _tag[j][0] = -1
        print "_tag: ", np.transpose(_tag)
        _w = lr(x, _tag)
        models[i] = _w
    print "train done"

def classify(x):
    _res = []
    _res.append(THETA(np.sum(np.dot(models[0], x))))
    _res.append(THETA(np.sum(np.dot(models[1], x))))
    _res.append(THETA(np.sum(np.dot(models[2], x))))
    return _res

def validate():
    print models
    for i in range(N):
        print "point index: ", i, '\t', np.array(classify(x[i])).argmax()

load()
ova()
validate()