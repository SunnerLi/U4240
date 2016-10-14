import math

dvc = 50.0
sigma = 0.05
E = 1

class Parrondo():
    N = 1000.0
    threshold = 10
    res = []    # To save the all eposon which smaller than the threshold
    diff = []   # To save the all difference which smaller than the threshold

    def __init__(self, N):
        self.N = N

    def Parrondo_left(self):
        return E

    def Parrondo_right(self):
        lnInside = 6 * mH(2*self.N) / sigma
        rootInside = ( 2 * E + math.log(lnInside) ) / self.N
        return pow(rootInside, 0.5)

    def Parrondo(self):
        global E
        E = 1
        
        while E >= 0:
            if math.fabs( self.Parrondo_left() - self.Parrondo_right() ) < self.threshold:
                self.res.append(E)
                self.diff.append(math.fabs( self.Parrondo_left() - self.Parrondo_right() ))
            E -= 0.0001
        
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


def mH(x):
    return pow(x, dvc-1)





#choose3 = Parrondo(100)
#print choose3.Parrondo()