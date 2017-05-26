import tensorflow as tf
from featureExtract import dataHolder
from tensorflow.contrib import layers
from tensorflow.contrib import losses

class baseClassifier:
	def __init__(self):
		self.data = dataHolder()
		self.timelength = 15
		self.num_features = 1
		self.lr_autoencoder = 1e-4
		self.lr_classifier = 1e-4
		self.batch_size = 100
		self.num_hidden = 100
		self.num_classes = 7
		self.batchType = "Predict single next"
		self.X = None
		self.Y = None
		self.YClass = None

	def addPlaceholders(self):
		if self.batchType == "Predict single next":
			self.X = tf.placeholder(tf.float32, [None, self.timelength, self.num_features])
			self.Y = tf.placeholder(tf.float32, [None, self.num_features])
			self.YClass = tf.placeholder(tf.int32, [None])
		else:
			raise NotImplementedError()

	def createModel(self):
		self.addPlaceholders()
		self.loss = None
		self.classificationLoss = None
		y_out = self.buildModel()
		self.loss, self.classificationLoss = self.addLoss(y_out)
		if self.loss == None or self.classificationLoss == None:
			raise "You need to set and return both the loss and the classification loss in function addLoss"
		self.optimizer, self.classificationOptimizer = self.addOptimizer()
		if self.optimizer == None or self.classificationOptimizer == None:
			raise "You need to set and return both the optimizer and the classification optimizer in function addLoss"
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


	def buildModel(self):
		raise NotImplementedError()

	def addLoss(self, y_out):
		raise NotImplementedError()

	def addOptimizer(self):
		raise NotImplementedError()

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






#p = baseClassifier()
#p.createModel()
#p.train()






