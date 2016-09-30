from homework15_IO import *
import numpy as np
import os
import sys

# Variable
D = []
X = []
Y = []

if __name__ == "__main__":
    D, X, Y = read() 

    res = pla(X, Y)
    print "Implement PLA, train from head again if wrong. The sum of epoch: ", res

    res = pla(X, Y, continue_if_wrong=True) 
    print "Implement PLA, still train to end if wrong. The sum of epoch: ", res