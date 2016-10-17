import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
import numpy as np

"""
    This program shows the single perceptron algorithm with tensorflow
    As the PLA's concept, one validation vector was sent to be checked each time.
    As the result, the batch size is 1. 
    At last, we would show the result of training.

    * In this program, the learning rate is 1.0
    * The picture is drawn by pyplot
    
    Author: SunnerLi
    Finish: 17/10/2016
"""
# variable
dimNumber = 3   # The dimention of the row (include bias)
rowNumber = 8   # The number of the whole row
batchSize = 1   # The validation size in each iteration

# x & y
data  = np.ndarray([rowNumber, dimNumber], dtype=float)
label = np.ndarray([rowNumber], dtype=float)

def read():
    """
        Read the file and change the contain as the corresponding format
    """
    global data
    global label
    count = 0      
    f = open('15_train_1.dat', 'r')
    while True:
        rawRow = f.readline().split(' ')
        if len(rawRow) == 1:
            break
        for i in xrange(dimNumber-1):
            data[count][i] = rawRow[i]
        data[count][2] = 0
        label[count] = rawRow[dimNumber-1]
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
        trainX = tf.placeholder(tf.float32, shape=[batchSize, dimNumber], name="X")
        trainY = tf.placeholder(tf.float32, shape=[batchSize], name="Y")

        # Define the weight and the flow process
        w = tf.Variable(tf.random_normal([dimNumber, 1]), dtype=tf.float32)
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
    iteration = 10
    with tf.Session(graph=graph) as session:
        init.run()
        for i in xrange(iteration):                 # The maximum iteration
            allRight = True                         # Initial checking flag

            # Test all training data
            for j in xrange(len(data)):             # The testing of each epoch
                X, Y = generateBatch(j)
                feed_dict = {trainX: X, trainY: Y}
                _op, _loss = session.run([optimizer, loss], feed_dict=feed_dict)
                if not _loss == 0.0:
                    allRight = False
                    update.eval(feed_dict=feed_dict)

            # print log
            if i % 500 == 0:
                print "w: ", w
                print "iter: ", i, "\tloss: ", _loss
            if allRight:
                break

        # Save the result
        print "Done, iter: ", i
        saver = tf.train.Saver()
        saver.save(session, 'perceptron_model.ckpt', global_step=iteration)

    # Validate
    predicts = []
    print("----- Validation -----")
    with tf.Session(graph=graph) as session:
        modelName = 'perceptron_model.ckpt-' + str(iteration)
        saver.restore(session, modelName)
        for i in range(rowNumber):
            xValid, yValid = generateBatch(i)
            feed_dict = {trainX: xValid, trainY: yValid}
            predicts.append(y.eval(feed_dict=feed_dict))
    draw(predicts)

def draw(_list):
    """
        Draw the image related to the predict result

        Arg:    The predicted result list
    """
    # Define multi-plots
    figure, (plot1, plot2) = plt.subplots(1, 2, sharey=True)

    # Draw origin plot
    for i in range(len(data)-1):
        if label[i] == 1:
            plot1.plot([data[i][0]], [data[i][1]], 'or', color='g')
        else:
            plot1.plot([data[i][0]], [data[i][1]], 'or', color='r') 
    plot1.set_xlim([-1, 5])
    plot1.set_ylim([-1, 5])
    plot1.set_title("Before Training")   

    # Draw result plot
    for i in range(len(data)-1):
        if _list[i] == 1:
            plot2.plot([data[i][0]], [data[i][1]], 'or', color='g')
        else:
            plot2.plot([data[i][0]], [data[i][1]], 'or', color='r')    
    plot2.set_xlim([-1, 5])
    plot2.set_ylim([-1, 5])
    plot2.set_title("After Training")

    # Show
    plt.show()

if __name__ == "__main__":
    read()
    work()