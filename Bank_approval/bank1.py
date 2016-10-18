import tensorflow as tf
import numpy as np

"""
    This program solve the bank approval problem by tensorflow
    This structure is multi-level proceptron which has 2 hidden layer

    Author: SunnerLi
    Finish: 18/10/2016
"""
# Yes or no convert
J = lambda x, y: "Yes" if x == y else "No"

# Tab macro to format the output 
TAB = lambda _list: '\t\t' if len(_list[0])+len(_list[1])+len(_list[2])+len(_list[3]) < 18 else '\t'

# Dictionary that convert the string to vector element
Age = {
    "young": 1, 
    "middle": 2, 
    "old": 3
}
Has_Job = {
    "false": 1, 
    "true": 2
}
Own_house = {
    "false": 1, 
    "true": 2
}
Credit_Rating = {
    "fair": 1, 
    "good": 2, 
    "excellent": 3
}
Approv = {
    "No": -1, 
    "Yes": 1
}

# Variable
dimNumber = 4           # The number of the feature in each row
rowNumber = 15          # The number of the whole row
h1_number = 8           # The size of 1-st hidden layer
h2_number = 4           # The size of 2-nd hidden layer
res_number = 1          # The size of result layer
iter = 1000             # The time of epoches
modelName = 'bank.cpkt' # The model name

# x & y
data  = np.ndarray([rowNumber, dimNumber], dtype=float)
label = np.ndarray([rowNumber], dtype=float) 

# Tensorflow layers difinition
weight = {
    'h1' : tf.Variable(tf.random_normal([dimNumber, h1_number])),
    'h2' : tf.Variable(tf.random_normal([h1_number, h2_number])),
    'out': tf.Variable(tf.random_normal([h2_number, res_number]))
}
bias = {
    'h1_b' : tf.Variable(tf.random_normal([h1_number])),
    'h2_b' : tf.Variable(tf.random_normal([h2_number])),
    'out_b': tf.Variable(tf.random_normal([res_number]))
}

def get(index):
    """
        Provide the x and y by the specific index

        Arg:    The index want to find
        Ret:    The specific x and y
    """
    _data = np.ndarray([1, dimNumber])
    _label = np.ndarray([1, 1])
    _data[0] = data[index]
    _label[0] = label[index]
    return _data, _label

def reverse(dictionary, value):
    """
        Return the key by the value toward the specific dictionary

        Arg:    The dictionary want to consult and value of the row
        Ret:    The target key
    """
    for key in dictionary:
        if dictionary[key] == value:
            return key

def read():
    """
        Preprocessing of the bank description file
    """
    count = 0
    with open('bank.txt', 'r') as f:
        while True:
            rawData = f.readline().split(' ')
            data[count][0] = Age[rawData[0]]
            data[count][1] = Has_Job[rawData[1]]
            data[count][2] = Own_house[rawData[2]]
            data[count][3] = Credit_Rating[rawData[3]]
            label[count] = Approv[rawData[4][:-1]]
            count += 1
            if count == rowNumber:
                break

def work():
    """
        Define, train and validate toward the structure
    """
    trainX = tf.placeholder(tf.float32, [1, dimNumber])
    trainY = tf.placeholder(tf.float32, [1, 1])

    # DNN difinition
    cc = tf.matmul(trainX, weight['h1'])
    fc1 = tf.add( tf.matmul(trainX, weight['h1']), bias['h1_b'])
    relu1 = tf.nn.relu(fc1)
    fc2 = tf.add( tf.matmul(relu1, weight['h2']), bias['h2_b'])
    relu2 = tf.nn.relu(fc2)
    outfc = tf.add( tf.matmul(relu2, weight['out']), bias['out_b'])
    out = tf.sign(outfc)

    # loss & optimizer
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(outfc, trainY)
    loss = tf.square(tf.reduce_sum(cross_entropy))
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    # init
    init = tf.initialize_all_variables()
    
    # Train
    with tf.Session() as session:
        # Initial the variable
        init.run()

        # Train for epoches
        for i in xrange(iter):
            lossSum = 0
            for j in xrange(rowNumber):
                x, y = get(j)
                feed_dict = {trainX: x, trainY: y}
                _, _loss = session.run([optimizer, loss], feed_dict=feed_dict)
                lossSum += _loss
                if not _loss == 0:
                    pass
            if i % 50 == 0:
                print "iter: ", i, "\tloss: ", lossSum
            if lossSum == 0:
                break

        # Save the result
        print "Finish training!"
        saver = tf.train.Saver()
        saver.save(session, modelName)

    # Validate
    print "----- Validation -----"
    with tf.Session() as session:
        init.run()
        saver.restore(session, modelName)
        for i in xrange(rowNumber):
            x, y = get(i)
            print "X:  ", recover(x), TAB(recover(x)),  # define below
            print "Y:  ", y, '\t',
            print "Y': ", out.eval(feed_dict={trainX: x, trainY: y}), '\t',
            print "Same?: ", J(y, out.eval(feed_dict={trainX: x, trainY: y}))
            
def recover(_list):
    """
        Recover the row vector to the description

        Arg:    The row vector
        Ret:    The description list
    """
    _list = _list[0]
    _res = [None] * 4
    _res[0] = reverse(Age, _list[0])
    _res[1] = reverse(Has_Job, _list[1])
    _res[2] = reverse(Own_house, _list[2])
    _res[3] = reverse(Credit_Rating, _list[3])
    return _res

if __name__ == "__main__":
    read()
    work()