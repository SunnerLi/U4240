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
    f = open('15_train.dat', 'r')
    while True:
        rawRow = f.readline().split(' ')
        if len(rawRow) == 1:
            break
        for i in xrange(dimNumber-1):
            data[count][i] = rawRow[i]
        data[count][3] = rawRow[3].split('\t')[0]
        label[count] = (rawRow[3].split('\t')[1]).split('\n')[0]
        #print "Data: ", data[count], '\t\tlabel: ', label[count]
        count += 1

def read_():
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
        _label = label[index]
        _data = np.asarray([ data[index] ])
    return _data, _label

def work():
    # Define network
    graph = tf.Graph()
    with graph.as_default():
        trainX = tf.placeholder(tf.float32, shape=[batchSize, dimNumber])
        trainY = tf.placeholder(tf.float32, shape=[batchSize])
        #w = tf.Variable(tf.random_normal([dimNumber, 1]), dtype=tf.float32)
        w = tf.Variable(np.asarray([[-1], [1], [-5]]), dtype=tf.float32)
        y = tf.sign(tf.matmul(trainX, w))
        loss = tf.reduce_mean(tf.square(tf.sub(y, trainY)))
        optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)


        mix = tf.add(w, tf.transpose(loss * trainX))
        update = tf.assign( w, mix)
        init = tf.initialize_all_variables()

    # Train
    iter = 1
    with tf.Session(graph=graph) as session:
        init.run()
        for i in xrange(iter):
            X, Y = generateBatch()
            feed_dict = {trainX: X, trainY: Y}
            #print w.eval()
            _op, _loss = session.run([optimizer, loss], feed_dict=feed_dict)

            # print log
            if i % 500 == 0:
                print "w: ", w
                print "mix: ", tf.shape(mix)
                print "iter: ", i, "\tloss: ", _loss
            if _loss < _error:
                break
        print "Done, iter: ", i

    # Validate
    predicts = []
    print("----- Validation -----")
    with tf.Session(graph=graph) as session:
        init.run()
        for i in range(rowNumber):
            xValid, yValid = generateBatch(i)
            print "X:  ", xValid
            print "Y:  ", yValid
            predicts.append(session.run([y], feed_dict={trainX: xValid}))
            print "Y': ", predicts[-1]
    draw(predicts)

def draw(_list):
    """
        Draw the image related to the predict result

        Arg:    The predicted result list
    """
    for i in range(len(data)-1):
        if _list[i][0][0][0] == 1:
            pylab.plot([data[i][0]], [data[i][1]], 'or', color='g')
        else:
            pylab.plot([data[i][0]], [data[i][1]], 'or', color='r')
    pylab.xlim([-1, 5])
    pylab.ylim([-1, 5])
    pylab.show()

read_()
generateBatch()
work()
#draw([1, 1, 1, 1, -1, -1, -1])