import tensorflow as tf
import numpy as np

"""
    This code solve the problem 20 by tensorflow
    But the answer is far from the correct answer of others
    So I might write another new one to vertify the answer
"""

# Training data variable
trainN = 1000
trainx = np.ndarray([trainN, 20])
trainy = np.ndarray([trainN, 1])

# Testing data variable
testN = 3000
testX = np.ndarray([testN, 20])
testY = np.ndarray([testN, 1])

# Training constants
iteration = 50000
eta = 0.001
batchSize = trainN

# The time period to show the loss information
showLossLimit = 500

if __name__ == "__main__":
    # Read the data
    with open('hw3_train.dat', 'r') as f:
        for i in range(trainN):
            string = f.readline().split()
            trainx[i] = np.asarray(string[:-1])
            trainy[i] = string[-1]
    with open('hw3_test.dat', 'r') as f:
        for i in range(testN):
            string = f.readline().split()
            testX[i] = np.asarray(string[:-1])
            testY[i] = string[-1]

    # Build the graph
    graph = tf.Graph()
    with graph.as_default():
        holderX = tf.placeholder(tf.float32, shape=[batchSize, 20])
        holderY = tf.placeholder(tf.float32, shape=[batchSize, 1])
        weight = tf.Variable(tf.random_normal([1, 20]), dtype=tf.float32)

        out = tf.matmul(holderX, tf.transpose(weight))
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(out, holderY))
        optimizer = tf.train.GradientDescentOptimizer(eta).minimize(loss)

        _0_1_error = tf.mul(holderY, tf.sign(out))
        init = tf.initialize_all_variables()

    # Tensorflow function
    with tf.Session(graph=graph) as session:
        # Train
        init.run()
        for i in range(iteration):
            _, _loss = session.run([optimizer, loss], feed_dict={holderX: trainx, holderY: trainy})
            if i % showLossLimit == 0:
                print "Iter: ", i, "\tloss: ", _loss
            if _loss < 0:
                break
        print "----- Q20 -----"

        # Validate
        valid = _0_1_error.eval(feed_dict={holderX: trainx, holderY: trainy})
        count = 0
        for i in range(batchSize):
            if valid[i] == -1:
                count += 1
        print "Ein: \t", float(count) / trainN

        # Test
        valid1 = _0_1_error.eval(feed_dict={holderX: testX[:1000], holderY: testY[:1000]})
        valid2 = _0_1_error.eval(feed_dict={holderX: testX[1000:2000], holderY: testY[1000:2000]})
        valid3 = _0_1_error.eval(feed_dict={holderX: testX[2000:], holderY: testY[2000:]})
        count = 0
        for i in range(batchSize):
            if valid1[i] == -1:
                count += 1
            if valid2[i] == -1:
                count += 1
            if valid3[i] == -1:
                count += 1
        print "Eout: \t", float(count) / testN