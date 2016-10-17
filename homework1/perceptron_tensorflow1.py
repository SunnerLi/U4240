from datetime import datetime
import tensorflow as tf
import numpy as np
import pylab

dimNumber = 3
rowNumber = 7
batchSize = 1
_error = 0.0001
biasNumber = 0

data  = np.ndarray([rowNumber, dimNumber], dtype=float)
label = np.ndarray([rowNumber], dtype=float)

def read():
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
        print "Data: ", data[count], '\t\tlabel: ', label[count]
        count += 1

def generateBatch(index=-1):
    np.random.seed(0)
    if index == -1:
        _data = np.random.choice(rowNumber, batchSize)
        _label = np.asarray([ label[x] for x in xrange(batchSize) ])
        _data = np.asarray([ data[x] for x in xrange(batchSize) ])
    else:
        _label = np.ndarray([batchSize], dtype=float)
        _label[0] = label[index]
        _data = np.ndarray([batchSize, dimNumber], dtype=float)
        _data[0] = data[index]
        print "shape: ", np.shape(_data)
    return _data, _label

def work():
    # Define network
    graph = tf.Graph()
    with graph.as_default():
        # Define the input and the label
        trainX = tf.placeholder(tf.float32, shape=[batchSize, dimNumber], name="X")
        trainY = tf.placeholder(tf.float32, shape=[batchSize], name="Y")

        # Define the weight and the flow process
        #w = tf.Variable(tf.random_normal([dimNumber, 1]), dtype=tf.float32)
        w = tf.Variable(np.asarray([[1], [1], [-5]]), dtype=tf.float32)
        y = tf.sign(tf.matmul(trainX, w))

        # Define the loss function and the optimization function
        loss = tf.reduce_mean(tf.sub(trainY, y))
        optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        # Define the update process
        mix = tf.transpose(loss * trainX)
        update = tf.assign(w, tf.add(w, mix))
        init = tf.initialize_all_variables()

    # Train
    iteration = 5
    with tf.Session(graph=graph) as session:
        init.run()
        for i in xrange(iteration):              # The maximum iteration
            allRight = True

            # Test all training data
            for j in xrange(len(data)):     # The testing of each epoch
                X, Y = generateBatch(j)
                print "X: ", X
                print "Y: ", Y
                feed_dict = {trainX: X, trainY: Y}
                _op, _loss = session.run([optimizer, loss], feed_dict=feed_dict)
                if not _loss == 0.0:
                    allRight = False
                    print "loss:", _loss
                    print "mix: ", mix.eval(feed_dict=feed_dict)
                    update.eval(feed_dict=feed_dict)
                    print "W': ", w.eval()
                else:
                    print "Y': ", y.eval(feed_dict=feed_dict)
                    print "No error"

            # print log
            if i % 500 == 0:
                print "w: ", w
                print "mix: ", tf.shape(mix)
                print "iter: ", i, "\tloss: ", _loss
            if allRight:
                break
        print "Done, iter: ", i

        print "final W: ", w.eval()
        saver = tf.train.Saver()
        saver.save(session, 'model.ckpt', global_step=iteration)

    # Validate
    predicts = []
    print("----- Validation -----")
    with tf.Session(graph=graph) as session:
        saver.restore(session, 'model.ckpt-5')
        print "init W: ", w.eval()
        for i in range(rowNumber):
            xValid, yValid = generateBatch(i)
            feed_dict = {trainX: xValid, trainY: yValid}
            print "X:  ", xValid
            print "Y:  ", yValid
            predicts.append(y.eval(feed_dict=feed_dict))
            print "Y': ", y.eval(feed_dict=feed_dict)
    draw(predicts)

def draw(_list):
    """
        Draw the image related to the predict result

        Arg:    The predicted result list
    """
    print "_list: ", _list
    for i in range(len(data)-1):
        if _list[i] == 1:
            pylab.plot([data[i][0]], [data[i][1]], 'or', color='g')
        else:
            pylab.plot([data[i][0]], [data[i][1]], 'or', color='r')
    pylab.xlim([-1, 5])
    pylab.ylim([-1, 5])
    pylab.show()

read()
generateBatch()
work()