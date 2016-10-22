import tensorflow as tf
import numpy as np
import collections
import random
import math

# Constant
MAX = 100000000
chooseNumber = 50000    # The number of the common words we want to choose
batch_size = 128
embedding_size = 128    # Dimension of the embedding vector.
skip_window = 1         # How many words to consider left and right.
num_skips = 2           # How many times to reuse an input to generate a label.
loss_sum  = 0           # The sum of the loss

# Validate constants
valid_size = 16         # The number of the words that are choosed to evaluate similarity
valid_window = 100      # The maximun length to choose for validation
valid_sample = np.random.choice(valid_window, valid_size, replace=False)
num_sample = 64         # The number of the words that are choose to evaluate the model

# The word data variables
words = list()            # The whole word list
freqs = None             # The frequency of each words
mapping = dict()        # word  -> index
batchIndex = 0          # Batch index to generate batch data



def buildDataset():
    """
    """
    global words
    global mapping
    global reverseMapping
    global freq    

    # Read the string & get the words
    f = open('text8', 'r')
    string = f.read()
    f.close()
    freqs = string.split(' ')
    freqs = collections.Counter(freqs).most_common(chooseNumber)
    freqs = [['UNDEFINE', 1]] + freqs

    # Build the mapping
    for word, freq in freqs:
        mapping[word] = len(mapping)

    # Build the words sequential
    for word in string.split(' '):
        if word in mapping:
            words.append(mapping[word])
        else:
            words.append(0)

def generateBatch(batchSize, repTime, relativeLength):
    """
        Generate the batch information according to the parameter

        Arg:    batchSize       - The number of the pair we should concern in each iteration
                repTime         - The repeated time that we concern for single word
                relativeLength  - The length to extend near the center
    """
    global batchIndex
    if repTime < 2 * relativeLength:
        print "-----> Error! <-----"
        return

    # Record the n-th words we want to consider in each iteration
    C = 2 * relativeLength + 1
    buf = collections.deque(maxlen=2*relativeLength+1)
    for i in range(2 * relativeLength + 1):
        buf.append( words[batchIndex] )
        batchIndex += 1
    #print buf

    # Generate the data and label
    data  = np.ndarray(shape=(batchSize), dtype=np.int32)
    label = np.ndarray(shape=(batchSize, 1), dtype=np.int32)
    for i in range(batchSize / repTime):
        center = relativeLength
        listUsed = [center]
        for j in range(repTime):
            # Choose a index that haven't been used
            while True:
                _ = random.randint(0, 2 * relativeLength)
                if _ not in listUsed:
                    listUsed.append(_)
                    break
            # Store the center and target
            #print "data[", i * repTime + j, "] = buf[", relativeLength, "] = ", buf[relativeLength]
            #print buf
            data[ i * repTime + j ] = buf[relativeLength]
            label[i * repTime + j ] = buf[_]

        # Update the buffer and index
        buf.append( words[batchIndex] )
        batchIndex += 1
    
    return data, label

def buildTensorflow():
    graph = tf.Graph()

    with graph.as_default():# set as default graph
        trainX = tf.placeholder(tf.int32, shape=[batch_size])
        trainY = tf.placeholder(tf.int32, shape=[batch_size, 1])

        with tf.device("/cpu:0"):
            # W
            embedding = tf.Variable(tf.random_normal([chooseNumber, batch_size], -1.0, 1.0))
            H = tf.nn.embedding_lookup(embedding, trainX)

            # W'
            #nsWeight = tf.Variable(tf.truncated_normal([batch_size, chooseNumber], -1.0, 1.0))
            nsWeight = tf.Variable( tf.truncated_normal( [chooseNumber, embedding_size], stddev=1.0 / math.sqrt(embedding_size) ) )
            nsBias = tf.Variable(tf.zeros([chooseNumber]))

        # Define the negation sampling loss function
        loss = tf.reduce_mean(tf.nn.nce_loss(nsWeight, nsBias, H, trainY, num_sample, chooseNumber))

        # Declare the optimize stradegy
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        
        init = tf.initialize_all_variables()

    # Start to train
    iteration = 5000
    with tf.Session(graph=graph) as session:
        init.run()
        print "--< Word Embedded>-- Initialize "

        for i in range(iteration):
            X, Y = generateBatch(batch_size, num_skips, skip_window)
            feed_dict = {trainX: X, trainY: Y}
            #print "iteration: ", i
            _, _loss = session.run([optimizer, loss], feed_dict=feed_dict)
            printLoss(i, _loss)
        tf.train.Saver().save(session, "model.ckpt", global_step=iteration)

def printLoss(iteration, loss):
    global loss_sum
    loss_sum += loss
    if iteration % 500 == 499:
        print "--< Word Embedded>-- Average loss: ", loss_sum / 2000
        loss_sum = 0

if __name__ == "__main__":
    buildDataset()
    #generateBatch(8, 2, 1)
    buildTensorflow()