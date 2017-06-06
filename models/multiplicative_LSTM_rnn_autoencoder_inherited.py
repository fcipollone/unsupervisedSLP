from base import baseClassifier
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from MultiplicativeLSTMCell import MultiplicativeLSTMCell

class multiplicative_LSTM_rnn_autoencoder_inherited(baseClassifier):
	def setBatchType(self):
		self.batchType = "vanilla autoencoder"

	def buildModel(self):
		batch_size = tf.shape(self.X)[0]
		self.myLSTM = MultiplicativeLSTMCell(self.num_hidden)
		#self.myGRU = tf.contrib.rnn.GRUCell(self.num_hidden,input_size=(None,self.timelength,self.num_features))
		outputs, _ = tf.nn.dynamic_rnn(self.myLSTM, self.X, initial_state = self.myLSTM.zero_state(batch_size,tf.float32), scope='step1/rnn1')
		self.secondLSTM = MultiplicativeLSTMCell(self.num_features)
		print self.myLSTM.state_size
		print self.secondLSTM.state_size
		#self.secondGRU = tf.contrib.rnn.GRUCell(self.num_features,input_size=(None,self.timelength,self.num_hidden))
		outputs, self.state = tf.nn.dynamic_rnn(self.secondLSTM, outputs, initial_state = self.secondLSTM.zero_state(batch_size,tf.float32),scope='step1/rnn2')
		self.state, _ = tf.split(self.state, [1,1], 0) # This is necessary when using the LSTM cell. splits tensor into sizes 1 and 1 on axis 0
		self.state = tf.reshape(self.state, [-1, self.num_features])

		#Self.state is my prediction for the next step, this next part is a fully connected layer from all hidden states
		#To the classification for part two.
		packedOutputs = tf.stack(outputs)
		reshapedOutputs = tf.reshape(packedOutputs, [-1, self.timelength*self.num_features])
		self.classification = tf.contrib.layers.fully_connected(reshapedOutputs, num_outputs=self.num_classes, weights_initializer = tf.contrib.layers.xavier_initializer(), scope='step2')
		return outputs

	def addLoss(self, y_out):
		l2_cost = 0
		for el in tf.trainable_variables():
			if 'weights:0' in el.name.split('/'):
				l2_cost += tf.nn.l2_loss(el)*.01
		self.lossWithoutReg = tf.reduce_mean(tf.losses.mean_squared_error(y_out,self.Y))*10000
		loss_summary = tf.summary.scalar('Autoencoder Loss without regularization', self.lossWithoutReg)

		self.secondLoss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.YClass, logits=self.classification))
		second_loss_summary = tf.summary.scalar('Classification Loss', self.secondLoss)
		return tf.reduce_mean(l2_cost) + self.lossWithoutReg, loss_summary, self.secondLoss, second_loss_summary

	def addAccuracy(self):
		self.secondAccuracy = tf.contrib.metrics.accuracy(labels=self.YClass, predictions=tf.to_int32(tf.argmax(self.classification,1)))
		accuracy_summary = tf.summary.scalar('Classification accuracy', self.secondAccuracy)

		return self.secondAccuracy, accuracy_summary

	def addOptimizer(self):
		optimizer = tf.train.AdamOptimizer(self.lr_autoencoder)
		step1Train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"step1")
		optimizer = optimizer.minimize(self.loss, var_list=step1Train)
		optimizer2 = tf.train.AdamOptimizer(self.lr_classifier)
		step2Train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"step2")
		classificationOptimizer = optimizer2.minimize(self.classificationLoss, var_list=step2Train)
		return optimizer, classificationOptimizer

