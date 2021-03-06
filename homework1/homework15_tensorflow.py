import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
import numpy as np

"""
    This program solves the problem 15 by tensorflow.
    As the PLA's concept, one validation vector was sent to be checked each time.
    As the result, the batch size is 1. 
    At last, we would show the result of training.

    * In this program, the bias is 1.0
    * Since 4-dimension cannot be project on 2D image, the plot below is just a rough demo
    
    Author: SunnerLi
    Finish: 18/10/2016
"""
# variable
featureNumber = 4               # The dimention of the row
dimNumber = featureNumber + 1   # The dimention of the row (include bias)
rowNumber = 400                 # The number of the whole row
batchSize = 1                   # The validation size in each iteration

# x & y
data  = np.ndarray([rowNumber, dimNumber], dtype=float)
label = np.ndarray([rowNumber], dtype=float)

def read():
    """
        Read the file and change the contain as the corresponding format
    """
    global data
    global label
    count = 0                               # counter index
    f = open('15_train.dat', 'r')
    while True:
        rawRow = f.readline().split(' ')
        if len(rawRow) == 1:
            break
        
        # Reformat the raw data
        data[count][0] = 1                  # bias = 1
        for i in xrange(featureNumber-1):   # don't consider the last feature since it's messy
            data[count][i+1] = rawRow[i]
        data[count][4] = rawRow[3].split('\t')[0]
        label[count] = (rawRow[3].split('\t')[1]).split('\n')[0]
        count += 1

def generateBatch(index):
    """
        Generate the batch data for the specific index

        Arg:    The index whose data want to be generated
        Ret:    The corresponding x & y
    """
    _label = np.ndarray([batchSize], dtype=float)
    _data = np.ndarray([batchSize, dimNumber], dtype=float)
    _label[0] = label[index]
    _data[0] = data[index]
    return _data, _label

def work():
    """
        Define the tensorflow and run the whole process
        The process includes training and validation
    """
    # Define network
    graph = tf.Graph()
    with graph.as_default():
        # Define the input and the label
        trainX = tf.placeholder(tf.float32, shape=[batchSize, dimNumber])
        trainY = tf.placeholder(tf.float32, shape=[batchSize])

        # Define the weight and the flow process
        w = tf.Variable(np.asarray([[4.5], [4.2481], [-1.887], [3.14], [5.69]]), dtype=tf.float32)
        y = tf.sign(tf.matmul(trainX, w))

        # Define the loss function and the optimization function
        loss = tf.reduce_mean(tf.sub(trainY, y))
        optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        # Define the update process
        revised = tf.transpose(loss * trainX)
        update = tf.assign(w, tf.add(w, revised))

        # Initial at last
        init = tf.initialize_all_variables()

    # Train
    iter = 50
    with tf.Session(graph=graph) as session:
        init.run()
        for i in xrange(iter):
            allRight = True                         # Initial checking flag

            # Test all training data
            lossSum = 0
            j = 0
            while True:                             # The testing of each epoch
                X, Y = generateBatch(j)
                feed_dict = {trainX: X, trainY: Y}
                _loss = session.run([loss], feed_dict=feed_dict)
                if not _loss == [0.0]:
                    allRight = False
                    lossSum += _loss[0]
                    update.eval(feed_dict=feed_dict)
                j += 1
                if j == rowNumber:
                    break               

            # print log
            print "w: ", w.eval()
            print "iter: ", i, "\tloss: ", lossSum
            if allRight:
                break
        
        # Save the result
        print "Done, iter: ", i
        saver = tf.train.Saver()
        saver.save(session, 'hw15_tensorflow.ckpt', global_step=iter)

    # Validate
    predicts = []
    print("----- Validation -----")
    with tf.Session(graph=graph) as session:
        init.run()
        modelName = 'hw15_tensorflow.ckpt-' + str(iter)
        saver = tf.train.Saver()
        saver.restore(session, modelName)
        for i in range(rowNumber):
            xValid, yValid = generateBatch(i)
            predicts.append(session.run([y], feed_dict={trainX: xValid})[0][0])
            #print "Y:  ", yValid, "\tY': ", y.eval(feed_dict={trainX: xValid})[0][0]
    draw(predicts)

def draw(_list):
    """
        Draw the image related to the predict result
        The x-coordinate is dim-1st while y-coordinate is dim-4th

        Arg:    The predicted result list
    """
    figure, (plot1, plot2) = plt.subplots(1, 2, sharex=True)

    # Plot the origin
    for i in range(len(data)-1):
        if label[i] == 1:
            plot1.plot([data[i][1]], [data[i][4]], 'or', color='g')
        else:
            plot1.plot([data[i][1]], [data[i][4]], 'or', color='r')
    plot1.set_title("Before Training")
    
    # Plot the result
    for i in range(len(data)-1):
        if _list[i] == 1:
            plot2.plot([data[i][1]], [data[i][4]], 'or', color='g')
        else:
            plot2.plot([data[i][1]], [data[i][4]], 'or', color='r')
    plot2.set_title("After Training")
    plt.show()

if __name__ == "__main__":
    read()
    work()