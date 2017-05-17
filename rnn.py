import tensorflow as tf
from featureExtract import dataHolder
from tensorflow.contrib import layers
from tensorflow.contrib import losses

class predictor:
	def __init__(self):
		self.data = dataHolder()
		self.timelength = 15
		self.num_features = 1
		self.lr = 1e-4
		self.batch_size = 20
		self.num_hidden = 100

		self.X = tf.placeholder(tf.float32, [None, self.timelength, self.num_features])
		self.Y = tf.placeholder(tf.float32, [None, self.num_features])

		self.loss = self.buildGraph()
		self.addOptimizer()

	def createFeedDict(self, X, Y):
		feedDict = {}
		feedDict[self.X] = X
		feedDict[self.Y] = Y
		return feedDict

	def buildGraph(self):
		y_out = self.buildModel()
		loss = self.addLoss(y_out)
		return loss

	def buildModel(self):
		batch_size = tf.shape(self.X)[0]
		self.myGRU = tf.contrib.rnn.GRUCell(self.num_hidden,input_size=(None,self.timelength,self.num_features))
		outputs, _ = tf.nn.dynamic_rnn(self.myGRU, self.X, initial_state = self.myGRU.zero_state(batch_size,tf.float32), scope='rnn1')
		self.secondGRU = tf.contrib.rnn.GRUCell(self.num_features,input_size=(None,self.timelength,self.num_hidden))
		outputs, self.state = tf.nn.dynamic_rnn(self.secondGRU, outputs, initial_state = self.secondGRU.zero_state(batch_size,tf.float32),scope='rnn2')
		return self.state

	def addLoss(self, y_out):
		l2_cost = 0
		for el in tf.trainable_variables():
			if 'weights:0' in el.name.split('/'):
				l2_cost += tf.nn.l2_loss(el)*.01
		self.lossWithoutReg = tf.reduce_mean(tf.losses.mean_squared_error(y_out,self.Y))*10000
		return tf.reduce_mean(l2_cost) + self.lossWithoutReg

	def addOptimizer(self):
		optimizer = tf.train.AdamOptimizer(self.lr)
		self.optimizer = optimizer.minimize(self.loss)

	def train(self):
		with tf.Session() as session:
			session.run(tf.global_variables_initializer())
			for i in range(10000):
				batch_x, batch_y = self.data.getBatchOf(size=self.batch_size, length=self.timelength)
				session.run([self.optimizer], feed_dict=self.createFeedDict(batch_x, batch_y))
				if i % 300 == 0:
					print ("Iteration: ", i)
					batch_x, batch_y = self.data.getBatchValid(size=self.batch_size, length=self.timelength)
					loss, lwr, yo = session.run([self.loss, self.lossWithoutReg, self.state], feed_dict=self.createFeedDict(batch_x, batch_y))
					print (yo)
					print ("Loss = ", loss)
					print ("Loss without regularization = ", lwr)
					print batch_x[0]

p = predictor()
p.train()






