import matplotlib.pyplot as plt
import plotin1 as pi1
import numpy as np

"""
    This program would generate three groups of data
    In each group, it would contain N data point
    So the total number of data point is 3N
"""

# The number of data you want to generate
N = 10

# Step 1. Generate the data
x1 = np.random.rand(N, 2)
x1[:, 0] *= 10
x1[:, 1] = x1[:, 1] * 10 + 6
y1 = np.ones([N]) - 1

x2 = np.random.rand(N, 2)
x2[:, 0] = x2[:, 0] * 10 + 10
x2[:, 1] *= 5
y2 = np.ones([N])

x3 = np.random.rand(N, 2)
x3[:, 0] *= 5
x3[:, 1] *= 5
y3 = np.ones([N]) + 1

x = np.round(np.concatenate((x1, x2, x3)), 2)
y = np.concatenate((y1, y2, y3))

# Step 2. Draw the distribution
pi1.PointChart(x[:, 0], x[:, 1], y).show()

# Step 3. Write into the file
string = ""
for i in range(N * 3):
    string = string + str(x[i][0]) + ',' + str(x[i][1]) + ',' + str(y[i]) + '\n'
with open('train.dat', 'w') as f:
    f.write(string)