from bigfloat import *
import math

dvc = 50.0
sigma = 0.05
E = 1

def mH(x):
    return pow(BigFloat(x), dvc-1)
    

class Devroye():
    N = 1000.0
    threshold = 0.9
    res = []    # To save the all eposon which smaller than the threshold
    diff = []   # To save the all difference which smaller than the threshold

    def __init__(self, N):
        self.N = N

    def Devroye_left(self):
        return E

    def Devroye_right(self):
        lnInside = 4 * BigFloat(mH( pow(self.N, 2) )) / sigma
        rootInside = ( 4 * E * ( 1 + E ) + math.log(lnInside) ) / ( 2 * self.N )
        return pow(rootInside, 0.5)

    def Devroye(self):
        global E
        E = 1
        
        while E >= 0:
            if math.fabs( self.Devroye_left() - self.Devroye_right() ) < self.threshold:
                self.res.append(E)
                self.diff.append(math.fabs( self.Devroye_left() - self.Devroye_right() ))
            E -= 0.001

        """
        for i in range(len(self.res)):
            print "E: ", self.res[i], "\tdirr: ", self.diff[i]
        """        
        
        # Count the minimum
        i = 0
        j = 1
        k = 2
        while True:
            if self.diff[i] > self.diff[j] and self.diff[k] > self.diff[j]:
                return self.res[j]
            elif k == len(self.res) - 1:
                return self.res[k]
            else:
                i += 1
                j += 1
                k += 1 
        

#devroye = Devroye(1000)
#print devroye.Devroye()