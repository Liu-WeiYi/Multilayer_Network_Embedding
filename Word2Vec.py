# Sailung Yeung
# <yeungsl@bu.edu>
# reference:
# http://www.cnblogs.com/edwardbi/p/5509699.html

import tensorflow as tf
import numpy as np
from six.moves import urllib
import collections, math, os, random, zipfile


class Learn:
# build data sets for training use
# different from the original version
# that it did not limit the vocabulary_size
# But limite the words who has relatively low frequency
  def __init__(self, words):
    self.W = words
    self.D = 0

  def build_dataset(self, words, too_low_freq):
  	### count -- word frequency list
  	### diciotnary -- word to int according to frequency
  	### reverse dictionary --  a look up map for output int back to word
  	### data -- used in tarining with int representation of word
  	 
  	count_org = [['UNK', -1]]
  	count_org.extend(collections.Counter(words).most_common())
  	count = [['UNK', -1]]
  	for word, c in count_org:
  		if c > too_low_freq:
  			count.append([word, c])
  	dictionary = dict()
  	for word, _ in count:
  		dictionary[word] = len(dictionary)
  	data = list()
  	unk_count = 0
  	for word in words:
  		if word in dictionary:
  			index = dictionary[word]
  		else:
  			index = 0
  			unk_count += 1
  		data.append(index)
  	count[0][1] = unk_count
  	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  	return data, count, dictionary, reverse_dictionary


  # Step 3: Function to generate a training batch for the skip-gram model.
  def generate_batch(self, data, batch_size, num_skips, skip_window):
    data_index = self.D
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
      buffer.append(data[data_index])
      data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
      target = skip_window  # target label at the center of the buffer
      targets_to_avoid = [skip_window]
      for j in range(num_skips):
        while target in targets_to_avoid:
          target = random.randint(0, span - 1)
        targets_to_avoid.append(target)
        batch[i * num_skips + j] = buffer[skip_window]
        labels[i * num_skips + j, 0] = buffer[target]
      buffer.append(data[data_index])
      data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    self.D = data_index
    return batch, labels


  def train(self):
    words = self.W
    data, count, dictionary, reverse_dictionary = self.build_dataset(words, 0)
    del words
    #print('Most common words', count[:5])
    #print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
    print('Length of the dictionary:', len(reverse_dictionary))

    batch, labels = self.generate_batch(data, batch_size=8, num_skips=2, skip_window=1)
    #for i in range(8):
      #print(batch[i], reverse_dictionary[batch[i]],
          #'->', labels[i, 0], reverse_dictionary[labels[i, 0]])

    data_index = 0

    # Step 4: Build and train a skip-gram model.
    vocabulary_size = len(reverse_dictionary)
    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 1       # How many words to consider left and right.
    num_skips = 2         # How many times to reuse an input to generate a label.

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 6     # Random set of words to evaluate similarity on.
    valid_window = 10  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_sampled = 6    # Number of negative examples to sample.

    graph = tf.Graph()

    with graph.as_default():

      # Input data.
      train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
      train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
      valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

      # Ops and variables pinned to the CPU because of missing GPU implementation
      with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

      # Compute the average NCE loss for the batch.
      # tf.nce_loss automatically draws a new sample of the negative labels each
      # time we evaluate the loss.
      loss = tf.reduce_mean(
          tf.nn.nce_loss(weights=nce_weights,
                         biases=nce_biases,
                         labels=train_labels,
                         inputs=embed,
                         num_sampled=num_sampled,
                         num_classes=vocabulary_size))

      # Construct the SGD optimizer using a learning rate of 1.0.
      optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

      # Compute the cosine similarity between minibatch examples and all embeddings.
      norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
      normalized_embeddings = embeddings / norm
      valid_embeddings = tf.nn.embedding_lookup(
          normalized_embeddings, valid_dataset)
      similarity = tf.matmul(
          valid_embeddings, normalized_embeddings, transpose_b=True)

      # Add variable initializer.
      init = tf.global_variables_initializer()

    # Step 5: Begin training.
    num_steps = 100001

    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
      init.run()
      #print("Initialized")

      average_loss = 0
      for step in range(num_steps):
        batch_inputs, batch_labels = self.generate_batch(data,
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
          if step > 0:
            average_loss /= 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          #print("Average loss at step ", step, ": ", average_loss)
          average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
          sim = similarity.eval()
          for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = "Nearest to %s:" % valid_word
            for k in range(top_k):
              close_word = reverse_dictionary[nearest[k]]
              log_str = "%s %s," % (log_str, close_word)
            #print(log_str)
      final_embeddings = normalized_embeddings.eval()
      print ("shape of the final embedding", final_embeddings.shape)
      return list(final_embeddings), dictionary


