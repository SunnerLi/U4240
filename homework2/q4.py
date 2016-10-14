from Parrondo import *
from Devroye import *
import math

N = 1000.0
dvc = 50.0
sigma = 0.05

def mH(x):
    return pow(x, dvc-1)

def VCBound():
    lnInside = 4 * mH(N) / sigma
    rootInside = (8 / N * math.log(lnInside))
    return pow(rootInside, 0.5)

def Rademacher():
    # Calculate the first item
    lnInside = 2 * N * mH(N)
    root1Inside = 2 / N * math.log(lnInside)
    item1 = pow(root1Inside, 0.5)

    # Calculate the second item
    root2Inside = 2 / N * math.log(1/sigma)
    item2 = pow(root2Inside, 0.5)

    # Calculate the third item
    item3 = 1 / N

    return item1 + item2 + item3

def VarientVCBound():
    lnInside = 2 * mH(N) / pow(sigma, 0.5)
    rootInside = 16 / N * math.log(lnInside)
    return pow(rootInside, 0.5)

if __name__ == "__main__":
    print "----->\tN = 1000"
    N = 1000.0
    par1 = Parrondo(1000)
    dev1 = Devroye(1000)
    print "Bound for origin VC Bound: ", VCBound()
    print "Bound for Rademacher: ", Rademacher()
    print "Bound for Parrondo: ", par1.Parrondo()
    print "Bound for Devroye: ", dev1.Devroye()
    print "Bound for varient VC bound: ", VarientVCBound()