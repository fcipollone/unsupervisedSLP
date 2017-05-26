from base import baseClassifier
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses

class rnn_inherited(baseClassifier):
	def buildModel(self):
		batch_size = tf.shape(self.X)[0]
		self.myLSTM = tf.contrib.rnn.BasicLSTMCell(self.num_hidden)
		#self.myGRU = tf.contrib.rnn.GRUCell(self.num_hidden,input_size=(None,self.timelength,self.num_features))
		outputs, _ = tf.nn.dynamic_rnn(self.myLSTM, self.X, initial_state = self.myLSTM.zero_state(batch_size,tf.float32), scope='step1/rnn1')
		self.secondLSTM = tf.contrib.rnn.BasicLSTMCell(self.num_features)
		print self.myLSTM.state_size
		print self.secondLSTM.state_size
		#self.secondGRU = tf.contrib.rnn.GRUCell(self.num_features,input_size=(None,self.timelength,self.num_hidden))
		outputs, self.state = tf.nn.dynamic_rnn(self.secondLSTM, outputs, initial_state = self.secondLSTM.zero_state(batch_size,tf.float32),scope='step1/rnn2')
		self.state, _ = tf.split(self.state, [1,1], 0) # This is necessary when using the LSTM cell. splits tensor into sizes 1 and 1 on axis 0
		self.state = tf.reshape(self.state, [-1,1])
		packedOutputs = tf.stack(outputs)
		#print packedOutputs.shape
		outputs = tf.reshape(packedOutputs, [-1, self.timelength*self.num_features])
		#print outputs.shape
		self.classification = tf.contrib.layers.fully_connected(outputs, num_outputs=self.num_classes, weights_initializer = tf.contrib.layers.xavier_initializer(), scope='step2')
		return self.state

	def addLoss(self, y_out):
		l2_cost = 0
		for el in tf.trainable_variables():
			if 'weights:0' in el.name.split('/'):
				l2_cost += tf.nn.l2_loss(el)*.01
		self.lossWithoutReg = tf.reduce_mean(tf.losses.mean_squared_error(y_out,self.Y))*10000
		tf.summary.scalar('Loss without regularization', self.lossWithoutReg)
		self.secondLoss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.YClass, logits=self.classification))
		return tf.reduce_mean(l2_cost) + self.lossWithoutReg, self.secondLoss


	def addOptimizer(self):
		optimizer = tf.train.AdamOptimizer(self.lr_autoencoder)
		step1Train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"step1")
		optimizer = optimizer.minimize(self.loss, var_list=step1Train)
		optimizer2 = tf.train.AdamOptimizer(self.lr_classifier)
		step2Train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"step2")
		classificationOptimizer = optimizer2.minimize(self.classificationLoss, var_list=step2Train)
		return optimizer, classificationOptimizer

p = rnn_inherited()
p.createModel()
p.train()

