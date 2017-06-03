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
		self.lr2 = 1e-4
		self.batch_size = 100
		self.num_hidden = 100
		self.num_classes = 7

		self.X = tf.placeholder(tf.float32, [None, self.timelength, self.num_features])
		self.Y = tf.placeholder(tf.float32, [None, self.num_features])
		self.YClass = tf.placeholder(tf.int32, [None])

		self.loss = self.buildGraph()
		self.addOptimizer()
		self.merged = tf.summary.merge_all()

	def createFeedDict(self, X, Y):
		feedDict = {}
		feedDict[self.X] = X
		feedDict[self.Y] = Y
		return feedDict

	def createFeedDict2(self, X, Y):
		feed_dict = {}
		feed_dict[self.X] = X
		feed_dict[self.YClass] = Y
		return feed_dict

	def buildGraph(self):
		y_out = self.buildModel()
		loss = self.addLoss(y_out)
		return loss

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
		return tf.reduce_mean(l2_cost) + self.lossWithoutReg

	def addOptimizer(self):
		optimizer = tf.train.AdamOptimizer(self.lr)
		step1Train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"step1")
		self.optimizer = optimizer.minimize(self.loss, var_list=step1Train)
		optimizer2 = tf.train.AdamOptimizer(self.lr2)
		step2Train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"step2")
		self.secondOptimizer = optimizer2.minimize(self.secondLoss, var_list=step2Train)

	def train(self):
		with tf.Session() as session:
			train_writer = tf.summary.FileWriter('train', session.graph)
			session.run(tf.global_variables_initializer())
			for i in range(10000):
				batch_x, batch_y = self.data.getBatchOf(size=self.batch_size, length=self.timelength)
				summary, _ = session.run([self.merged, self.optimizer], feed_dict=self.createFeedDict(batch_x, batch_y))
				train_writer.add_summary(summary, i)
				if i % 300 == 0:
					print ("Iteration: ", i)
					batch_x, batch_y = self.data.getBatchValid(size=self.batch_size, length=self.timelength)
					loss, lwr, yo = session.run([self.loss, self.lossWithoutReg, self.state], feed_dict=self.createFeedDict(batch_x, batch_y))
					print ("Loss = ", loss)
					print ("Loss without regularization = ", lwr)
			for i in range(10000):
				batch_x, batch_y = self.data.getBatchWithLabels(size=self.batch_size, length=self.timelength)
				session.run([self.secondOptimizer], feed_dict=self.createFeedDict2(batch_x, batch_y))
				if i % 100 == 0:
					print ("Iteration: ", i)
					batch_x, batch_y = self.data.getBatchWithLabelsValid(size=self.batch_size, length=self.timelength)
					loss = session.run([self.secondLoss], feed_dict=self.createFeedDict2(batch_x, batch_y))
					print ("Loss = ", loss)
					#print ("Loss without regularization = ", lwr)


p = predictor()
p.train()






