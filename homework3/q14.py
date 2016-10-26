from q13 import *

# Generate points
X = (np.random.rand(N, 3) - 0.5) * 2
Y = np.ndarray([N, 1])
X[0][:] = 1
W = None

# Generate tags
for i in range(N):
    Y[i][0] = f(X[i][1], X[i][2])
    if random.randint(0, 9) == 0:
        Y[i][0] = -Y[i][0]

# Transform to the 7-dim space
_X = np.ndarray([N, 6])

print X[:][1]
print X[1][:]

_X[:][0] = 1
_X[:][1] = X[:][1]
_X[:][0] = X[:][2]
_X[:][0] = np.mat(X[:][1], X[:][2])
_X[:][0] = np.square(X[:][1])
_X[:][0] = np.square9X[:][2]