from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.datasets import imdb

num_words = 1000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = num_words)

X_train = sequence.pad_sequences(X_train, maxlen=200)
X_test = sequence.pad_sequences(X_test, maxlen=200)


# Define network architecture and compile
model = Sequential() 
model.add(Embedding(num_words, 50, input_length=200)) 
model.add(Dropout(0.2)) 
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2)) 
model.add(Dense(250, activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train,y_train, epochs=2, batch_size=150)
predict = model.predict(X_test)

print(y_test,predict)
import numpy as np
pred =np.array([1 if p>=0.5 else 0 for p in predict])

from sklearn import metrics
print(metrics.classification_report(y_test,pred))










'''
import numpy as np
import collections


class DataHandler:
	def read_data(self, fname):
		with open(fname) as f:
			content = f.readlines()
		content = [x.strip() for x in content]
		content = [content[i].split() for i in range(len(content))]
		content = np.array(content)
		content = np.reshape(content, [-1, ])
		return content

	def build_datasets(self, words):
		count = collections.Counter(words).most_common()
		dictionary = dict()
		for word, _ in count:
			dictionary[word] = len(dictionary)
		reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
		return dictionary, reverse_dictionary


import tensorflow as tf
from tensorflow.contrib import rnn


class RNNGenerator:
	def create_LSTM(self, inputs, weights, biases, seq_size, num_units):
		# Reshape input to [1, sequence_size] and split it into sequences
		inputs = tf.reshape(inputs, [-1, seq_size])
		inputs = tf.split(inputs, seq_size, 1)

		# LSTM with 2 layers
		rnn_model = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_units), rnn.BasicLSTMCell(num_units)])

		# Generate prediction
		outputs, states = rnn.static_rnn(rnn_model, inputs, dtype=tf.float32)

		return tf.matmul(outputs[-1], weights['out']) + biases['out']



import tensorflow as tf
import random
import numpy as np


class SessionRunner():
	training_iters = 50000

	def __init__(self, optimizer, accuracy, cost, lstm, initilizer, writer):
		self.optimizer = optimizer
		self.accuracy = accuracy
		self.cost = cost
		self.lstm = lstm
		self.initilizer = initilizer
		self.writer = writer

	def run_session(self, x, y, n_input, dictionary, reverse_dictionary, training_data):

		with tf.Session() as session:
			session.run(self.initilizer)
			step = 0
			offset = random.randint(0, n_input + 1)
			acc_total = 0

			self.writer.add_graph(session.graph)

			while step < self.training_iters:
				if offset > (len(training_data) - n_input - 1):
					offset = random.randint(0, n_input + 1)

				sym_in_keys = [[dictionary[str(training_data[i])]]
				               for i in range(offset, offset + n_input)]
				sym_in_keys = np.reshape(np.array(sym_in_keys), [-1, n_input, 1])

				sym_out_onehot = np.zeros([len(dictionary)], dtype=float)
				sym_out_onehot[dictionary[str(training_data[offset + n_input])]] = 1.0
				sym_out_onehot = np.reshape(sym_out_onehot, [1, -1])

				_, acc, loss, onehot_pred = session.run([self.optimizer, self.accuracy,
				                                         self.cost, self.lstm],
				                                        feed_dict={x: sym_in_keys, y: sym_out_onehot})
				acc_total += acc

				if (step + 1) % 1000 == 0:
					print("Iteration = " + str(step + 1) + ", Average Accuracy= " +
					      "{:.2f}%".format(100 * acc_total / 1000))
					acc_total = 0
				step += 1
				offset += (n_input + 1)

import tensorflow as tf
# from DataHandler import DataHandler
# from RNN_generator import RNNGenerator
# from session_runner import SessionRunner

log_path = '/output/tensorflow/'
writer = tf.summary.FileWriter(log_path)

# Load and prepare data
data_handler = DataHandler()

training_data =  data_handler.read_data('meditations.txt')

dictionary, reverse_dictionary = data_handler.build_datasets(training_data)

# TensorFlow Graph input
n_input = 3
n_units = 512

x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, len(dictionary)])

# RNN output weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_units, len(dictionary)]))
}
biases = {
    'out': tf.Variable(tf.random_normal([len(dictionary)]))
}

rnn_generator = RNNGenerator()
lstm = rnn_generator.create_LSTM(x, weights, biases, n_input, n_units)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=lstm, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(lstm,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
initilizer = tf.global_variables_initializer()

session_runner = SessionRunner(optimizer, accuracy, cost, lstm, initilizer, writer)
session_runner.run_session(x, y, n_input, dictionary, reverse_dictionary, training_data)

'''