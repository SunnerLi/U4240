from __future__ import division
from collections import Counter, defaultdict
from random import shuffle
import tensorflow as tf
import numpy as np
import cPickle
import os

"""
    The core implementation of the GloVe word model

    Revise: SunnerLi (Not first author)
    Finish: 23/10/2016
"""

vocabLimit = 5000000        # The number of limit that should save the period result
mixTime = 5                 # The repeat time that want to eliminate the influence of seperation (Should >= 1)
cooccurrence_counts = None

class NotTrainedError(Exception):
    pass

class NotFitToCorpusError(Exception):
    pass

class FileNotFoundError(Exception):
    """
        If the .pkl file isn't exist, it would be raised
    """
    pass

class SaveFileNotSuccessfulError(Exception):
    """
        If the .pkl file didn't store correctly, it would be raised
    """
    pass

class GloVeModel():
    def __init__(self, embedding_size, context_size, max_vocab_size=100000, min_occurrences=1,
                 scaling_factor=3.0/4.0, cooccurrence_cap=100, batch_size=512, learning_rate=0.05):
        self.embedding_size = embedding_size
        if isinstance(context_size, tuple):
            self.left_context, self.right_context = context_size
        elif isinstance(context_size, int):
            self.left_context = self.right_context = context_size
        else:
            raise ValueError("`context_size` should be an int or a tuple of two ints")
        self.max_vocab_size = max_vocab_size
        self.min_occurrences = min_occurrences
        self.scaling_factor = scaling_factor
        self.cooccurrence_cap = cooccurrence_cap
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.__words = None
        self.__word_to_id = None
        self.__cooccurrence_matrix = None
        self.__embeddings = None
        self.__wantSplit = False
        self.__numberOfSplitOcc = 0

    def fit_to_corpus(self, corpus, split=False):
        """
            Deal with the corpus

            Arg:    corpus  - The list of the corpus vocabularies
                    split   - If you want to split the result (low-ability computer suggest)
        """
        self.__wantSplit = split
        self.__fit_to_corpus(corpus, self.max_vocab_size, self.min_occurrences, self.left_context, self.right_context)
        #self.__build_graph()

    def __fit_to_corpus(self, corpus, vocab_size, min_occurrences, left_size, right_size):
        """
            Statistic the frequency and store the result

            Arg:    corpus          - The list of the corpus vocabularies
                    vocab_size      - The maximun number of common vocabulary want to choose
                    min_occurrences - if the would didn't appear over this limit, ignore it
                    left_size       - The M/2 on the left (half window size)
                    right_size      - The M/2 on the right (half window size)
        """
        # Initialize
        global cooccurrence_counts
        word_counts = Counter()
        cooccurrence_counts = defaultdict(float)
        
        # Scan the corpus and statistic the frequency
        for region in corpus:
            word_counts.update(region)
            countAlpha = 0
            for l_context, word, r_context in _context_windows(region, left_size, right_size):
                if countAlpha % 10000 == 0:
                    print "<GloVe>-- ", "Number of words isn't scanned: ", len(region) - countAlpha
                for i, context_word in enumerate(l_context[::-1]):
                    # add (1 / distance from focal word) for this pair
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
                for i, context_word in enumerate(r_context):
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
                countAlpha += 1
        if len(cooccurrence_counts) == 0:
            raise ValueError("No coccurrences in corpus. Did you try to reuse a generator?")
        self.__words = [word for word, count in word_counts.most_common(vocab_size)
                        if count >= min_occurrences]
        self.__word_to_id = {word: i for i, word in enumerate(self.__words)}

        # Split(store) the data or not
        if self.__wantSplit == True:
            if not os.path.isdir('tmp'):
                os.mkdir('tmp')
            cPickle.dump(self.__words, open("./tmp/glove_words.pkl", 'w'))
            cPickle.dump(self.__word_to_id, open("./tmp/glove_id.pkl", 'w'))
            #cPickle.dump(cooccurrence_counts, open("./tmp/glove_count.pkl", 'w'))
            print "<GloVe>-- ", "Write done"
            #cooccurrence_counts = None
        else:
            print "<GloVe>-- ", "Co-occurence size: ", len(cooccurrence_counts)
            self.__cooccurrence_matrix = {
                (self.__word_to_id[words[0]], self.__word_to_id[words[1]]): count
                for words, count in cooccurrence_counts.items()
                if words[0] in self.__word_to_id and words[1] in self.__word_to_id}     

    def buils_matrix(self):
        """
            Build the co-occurence matrix
        """
        if os.path.isdir('tmp'):
            if os.path.isfile("./tmp/glove_words.pkl") and os.path.isfile("./tmp/glove_id.pkl") and os.path.isfile("./tmp/glove_count.pkl"):
                # Load the statistic element
                if self.__words == None:
                    self.__words = cPickle.load(open("./tmp/glove_words.pkl", 'r'))
                if self.__word_to_id == None:
                    self.__word_to_id = cPickle.load(open("./tmp/glove_id.pkl", 'r'))
                #cooccurrence_counts = cPickle.load(open("./tmp/glove_count.pkl", 'r'))
                print "<GloVe>-- ", "Co-occurence size: ", len(cooccurrence_counts)

                # Statistic the frequency
                self.__cooccurrence_matrix = dict()
                storeIndex = 0
                for words, count in cooccurrence_counts.items():
                    #print "<GloVe>-- ", "Statistic index: ", storeIndex
                    # If the both word is represent, then record the element
                    if words[0] in self.__word_to_id and words[1] in self.__word_to_id:
                        _key = (self.__word_to_id[words[0]], self.__word_to_id[words[1]])
                        self.__cooccurrence_matrix[_key] = count

                        # Clear the whole matrix if it's over limit
                        if len(self.__cooccurrence_matrix) >= vocabLimit:
                            _fileName = "./tmp/glove_occ.pkl-" + str(storeIndex)
                            print "<GloVe>-- ", "Save " + _fileName
                            cPickle.dump(self.__cooccurrence_matrix, open(_fileName, 'w'))
                            self.__cooccurrence_matrix = dict()
                            if not os.path.isfile(_fileName):
                                raise SaveFileNotSuccessfulError("didn't save ", _fileName)
                            storeIndex += 1

                    # Print the progress
                    if len(self.__cooccurrence_matrix) % 500000 == 0:
                        print "<GloVe>-- ", "Split part: ", storeIndex, \
                            "\tdict number: ", len(self.__cooccurrence_matrix), "Rest: ", \
                            max(0, len(cooccurrence_counts) - (storeIndex) * 5000000 - len(self.__cooccurrence_matrix))

                # Store the last
                print "<GloVe>-- ", "Save glove_occ.pkl-"+ str(storeIndex)
                cPickle.dump(self.__cooccurrence_matrix, open("./tmp/glove_occ.pkl-" + str(storeIndex), 'w'))
                self.__cooccurrence_matrix = dict()
                self.__numberOfSplitOcc = storeIndex+1
            else:
                raise FileNotFoundError("No found the split file, try to train again!")
        else:
            raise FileNotFoundError("No found the tmp folder, try to train again!")


    def __build_graph(self):
        """
            Build the tensorflow graph
        """
        self.__graph = tf.Graph()
        with self.__graph.as_default(), self.__graph.device(_device_for_node):
            count_max = tf.constant([self.cooccurrence_cap], dtype=tf.float32,
                                    name='max_cooccurrence_count')
            scaling_factor = tf.constant([self.scaling_factor], dtype=tf.float32,
                                         name="scaling_factor")
            
            # Declare place holder
            print "<GloVe>-- ", "build placeholder"           
            self.__focal_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                name="focal_words")
            self.__context_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                  name="context_words")
            self.__cooccurrence_count = tf.placeholder(tf.float32, shape=[self.batch_size],
                                                       name="cooccurrence_count")

            # Declare weight and bias
            print "<GloVe>-- ", "build weight"
            focal_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0),
                name="focal_embeddings")
            context_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0),
                name="context_embeddings")
            focal_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0),
                                       name='focal_biases')
            context_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0),
                                         name="context_biases")

            # Declare word embedded wrapper
            focal_embedding = tf.nn.embedding_lookup([focal_embeddings], self.__focal_input)
            context_embedding = tf.nn.embedding_lookup([context_embeddings], self.__context_input)
            focal_bias = tf.nn.embedding_lookup([focal_biases], self.__focal_input)
            context_bias = tf.nn.embedding_lookup([context_biases], self.__context_input)

            # Declare loss and optimizer
            weighting_factor = tf.minimum(
                1.0,
                tf.pow(
                    tf.div(self.__cooccurrence_count, count_max),
                    scaling_factor))

            embedding_product = tf.reduce_sum(tf.mul(focal_embedding, context_embedding), 1)

            log_cooccurrences = tf.log(tf.to_float(self.__cooccurrence_count))

            distance_expr = tf.square(tf.add_n([
                embedding_product,
                focal_bias,
                context_bias,
                tf.neg(log_cooccurrences)]))

            single_losses = tf.mul(weighting_factor, distance_expr)
            self.__total_loss = tf.reduce_sum(single_losses)
            tf.scalar_summary("GloVe loss", self.__total_loss)
            self.__optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(
                self.__total_loss)
            self.__summary = tf.merge_all_summaries()

            self.__combined_embeddings = tf.add(focal_embeddings, context_embeddings,
                                                name="combined_embeddings")

    def train(self, num_epochs, log_dir=None, summary_batch_interval=1000,
              tsne_epoch_interval=None, savePath=None):
        """
            Train model interface

            Arg:    num_epochs  - The number of training epochs
                    savePath    - The save path of the embedded
        """
        # Check if we want to split training
        _fileList = os.listdir('./tmp/')
        _count = 0
        for _fileName in _fileList:
            if _fileName[:14] == "glove_occ.pkl-":
                _count += 1
        if not _count == 0:
            self.__wantSplit = True
            self.__numberOfSplitOcc = _count

        # Load training element and build the graph
        if self.__words == None:
            self.__words = cPickle.load(open("./tmp/glove_words.pkl", 'r'))
        if self.__word_to_id == None:
            self.__word_to_id = cPickle.load(open("./tmp/glove_id.pkl", 'r'))
        self.__build_graph()

        # Train
        print "<GloVe>-- ", "----- Training -----"
        self._train(num_epochs, log_dir, summary_batch_interval, tsne_epoch_interval, savePath)

    def _train(self, num_epochs, log_dir=None, summary_batch_interval=1000,
              tsne_epoch_interval=None, savePath=None):
        """
            Train the word embedded

            Arg:    num_epochs  - The number of training epochs
                    savePath    - The save path of the embedded
        """
        global mixTime
        
        with tf.Session(graph=self.__graph) as session:
            tf.initialize_all_variables().run()

            # If we want to seperate or not
            if self.__wantSplit == False:
                mixTime = 1
            for j in range(mixTime):
                print "<GloVe>-- ", "----- Mix epoch: ", j
                for i in range(self.__numberOfSplitOcc):
                    if self.__wantSplit == True:
                        print "<GloVe>-- ", "----- Load Dataset ", i
                        self.__cooccurrence_matrix = cPickle.load(open("./tmp/glove_occ.pkl-"+str(i), 'r'))

                    should_write_summaries = log_dir is not None and summary_batch_interval
                    should_generate_tsne = log_dir is not None and tsne_epoch_interval
                    batches = self.__prepare_batches()
                    total_steps = 0
                
                    if should_write_summaries:
                        summary_writer = tf.train.SummaryWriter(log_dir, graph_def=session.graph_def)
                    
                    # For loop training
                    for epoch in range(num_epochs):
                        shuffle(batches)
                        for batch_index, batch in enumerate(batches):
                            i_s, j_s, counts = batch
                            if len(counts) != self.batch_size:
                                continue
                            feed_dict = {
                                self.__focal_input: i_s,
                                self.__context_input: j_s,
                                self.__cooccurrence_count: counts}
                            session.run([self.__optimizer], feed_dict=feed_dict)
                            if should_write_summaries and (total_steps + 1) % summary_batch_interval == 0:
                                summary_str = session.run(self.__summary, feed_dict=feed_dict)
                                summary_writer.add_summary(summary_str, total_steps)
                            total_steps += 1
                        print "<GloVe>-- ", "Epoch: ", epoch, "\tloss: ", \
                            session.run(self.__total_loss, feed_dict=feed_dict)
                        if should_generate_tsne and (epoch + 1) % tsne_epoch_interval == 0:
                            current_embeddings = self.__combined_embeddings.eval()
                            output_path = os.path.join(log_dir, "epoch{:03d}.png".format(epoch + 1))
                            self.generate_tsne(output_path, embeddings=current_embeddings)
                    self.__embeddings = self.__combined_embeddings.eval()
                    if should_write_summaries:
                        summary_writer.close()

            if not savePath == None:    
                cPickle.dump(self.__embeddings, open(savePath, 'wb'))

    def load(self, modelPath):
        """
            Load the word embedded
            This function can be called individually if you had store the .pkl file

            Arg:    modelPath   - The path of the model which you want to load
        """
        # Load training element and build the graph
        if self.__words == None:
            self.__words = cPickle.load(open("./tmp/glove_words.pkl", 'r'))
        if self.__word_to_id == None:
            self.__word_to_id = cPickle.load(open("./tmp/glove_id.pkl", 'r'))
        self.__build_graph()

        # Load word embedded
        self.__embeddings = cPickle.load(open(modelPath, 'rb'))

    def cosSim(self, word_str1_or_id1, word_str2_or_id2):
        """
            Return the cosine similarity of the two words

            Arg:    word_str1_or_id1    - The word or the id of the first word
                    word_str2_or_id2    - The word or the id of the second word
        """
        vec1 = self.embedding_for(word_str1_or_id1)
        vec2 = self.embedding_for(word_str2_or_id2)

        # Compute the length of the vector 1
        length1 = np.sqrt( np.sum( np.square(vec1) ) )
        length2 = np.sqrt( np.sum( np.square(vec2) ) )
        return np.dot(vec1, vec2) / (length1 * length2) 

    def embedding_for(self, word_str_or_id):
        """
            Return the word vector of the specific word

            Arg:    The word or the id of the word
            Ret:    The ndarray vector
        """
        if isinstance(word_str_or_id, str):
            return self.embeddings[self.__word_to_id[word_str_or_id]]
        elif isinstance(word_str_or_id, int):
            return self.embeddings[word_str_or_id]

    def __prepare_batches(self):
        if self.__cooccurrence_matrix is None:
            raise NotFitToCorpusError(
                "Need to fit model to corpus before preparing training batches.")
        cooccurrences = [(word_ids[0], word_ids[1], count)
                         for word_ids, count in self.__cooccurrence_matrix.items()]
        i_indices, j_indices, counts = zip(*cooccurrences)
        return list(_batchify(self.batch_size, i_indices, j_indices, counts))

    @property
    def vocab_size(self):
        return len(self.__words)

    @property
    def words(self):
        if self.__words is None:
            raise NotFitToCorpusError("Need to fit model to corpus before accessing words.")
        return self.__words

    @property
    def embeddings(self):
        if self.__embeddings is None:
            raise NotTrainedError("Need to train model before accessing embeddings")
        return self.__embeddings

    def id_for_word(self, word):
        if self.__word_to_id is None:
            raise NotFitToCorpusError("Need to fit model to corpus before looking up word ids.")
        return self.__word_to_id[word]

    def generate_tsne(self, path=None, size=(100, 100), word_count=1000, embeddings=None):
        if embeddings is None:
            embeddings = self.embeddings
        from sklearn.manifold import TSNE
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs = tsne.fit_transform(embeddings[:word_count, :])
        labels = self.words[:word_count]
        return _plot_with_labels(low_dim_embs, labels, path, size)


def _context_windows(region, left_size, right_size):
    for i, word in enumerate(region):
        start_index = i - left_size
        end_index = i + right_size
        left_context = _window(region, start_index, i - 1)
        right_context = _window(region, i + 1, end_index)
        yield (left_context, word, right_context)


def _window(region, start_index, end_index):
    """
        Returns the list of words starting from `start_index`, going to `end_index`
        taken from region. If `start_index` is a negative number, or if `end_index`
        is greater than the index of the last word in region, this function will pad
        its return value with `NULL_WORD`.
    """
    last_index = len(region) + 1
    selected_tokens = region[max(start_index, 0):min(end_index, last_index) + 1]
    return selected_tokens


def _device_for_node(n):
    if n.type == "MatMul":
        return "/gpu:0"
    else:
        return "/cpu:0"


def _batchify(batch_size, *sequences):
    for i in xrange(0, len(sequences[0]), batch_size):
        yield tuple(sequence[i:i+batch_size] for sequence in sequences)


def _plot_with_labels(low_dim_embs, labels, path, size):
    import matplotlib.pyplot as plt
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    figure = plt.figure(figsize=size)  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right',
                     va='bottom')
    if path is not None:
        figure.savefig(path)
        plt.close(figure)
