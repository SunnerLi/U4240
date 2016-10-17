import tensorflow as tf
import numpy as np

dimNumber = 2
rowNumber = 7
batchSize = 7
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
        for i in xrange(dimNumber):
            data[count][i] = rawRow[i]
        label[count] = rawRow[dimNumber]
        print "Data: ", data[count], '\t\tlabel: ', label[count]
        count += 1

def generateBatch():
    _data = np.random.choice(rowNumber, batchSize)
    _label = np.asarray([ label[x] for x in xrange(batchSize) ])
    _data = np.asarray([ data[x] for x in xrange(batchSize) ])
    return _data, _label

def work():
    # Define network
    graph = tf.Graph()
    with graph.as_default():
        trainX = tf.placeholder(tf.float32, shape=[batchSize, dimNumber])
        trainY = tf.placeholder(tf.float32, shape=[batchSize])
        b = tf.constant(biasNumber, dtype=tf.float32)
        #w = tf.Variable(tf.random_normal([dimNumber, 1]))
        w = tf.Variable(np.asarray([[0.5], [0.5]]), dtype=tf.float32)
        y = tf.add( tf.matmul(trainX, w), b)
        error = tf.sub(y, trainY)
        loss = tf.reduce_mean(tf.square(error))
        #optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        init = tf.initialize_all_variables()

    # Train
    iter = 10000
    with tf.Session(graph=graph) as session:
        init.run()
        for i in xrange(iter):
            X, Y = generateBatch()
            feed_dict = {trainX: X, trainY: Y}
            _op, _loss = session.run([optimizer, loss], feed_dict=feed_dict)

            # print log
            if i % 500 == 0:
                print "iter: ", i, "\tloss: ", _loss
            if _loss < _error:
                break
        print "Done, iter: ", i

    # Validate
    print("----- Validation -----")
    with tf.Session(graph=graph) as session:
        init.run()
        xValid, yValid = generateBatch()
        print "X:  ", xValid
        print "Y:  ", yValid
        print "Y': ", session.run([y], feed_dict={trainX: xValid})[0]


read_()
generateBatch()
work()