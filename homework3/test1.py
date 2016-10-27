import numpy as np

a = np.asarray([[1, 2, 3], [2, 3, 4]])
#print np.asarray([ a[i][1] for i in range(2) ])
print a[:, 1]
print a[1][:]