import tensorflow as tf
from featureExtract import dataHolder
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from time import gmtime, strftime
import numpy as np
import os

class baseClassifier:
	def __init__(self, FLAGS):
		self.FLAGS = FLAGS
		self.data = dataHolder(FLAGS)
		self.timelength = FLAGS.time_length
		self.num_features = len(self.FLAGS.indices)
		self.lr_autoencoder = FLAGS.autoencoder_learning_rate
		self.lr_classifier = FLAGS.classifier_learning_rate
		self.iterations_autoencoder = FLAGS.iterations_autoencoder
		self.iterations_classification = FLAGS.iterations_classifier
		self.batch_size = FLAGS.batch_size
		self.num_hidden = 100
		self.num_classes = FLAGS.num_classes
		self.train_autoencoder = FLAGS.train_autoencoder
		self.train_classifier = FLAGS.train_classifier
		self.compute_validation_accuracy = FLAGS.validation_accuracy
		self.compute_test_accuracy = FLAGS.test_accuracy

		#Batch type is a feature because I want to allow for multiple ways to input data
		#	1) Currently "Predict single next" is the only one -- you get 'timelength' previous steps
		#	and need to predict the next step's features.
		#	2) Another possible way would be to predict the next step's features for every timestep.
		#	In other words you'd feed 15 timesteps in for X, and 15 in for Y, but the 15 for Y would be
		#	offset by one.
		self.batchType = "Predict single next"
		self.setBatchType()
		self.X = None
		self.Y = None
		self.YClass = None

	def setBatchType(self):
		self.batchType = "Predict single next"

	def addPlaceholders(self):
		#Pretty standard. 
		if self.batchType == "Predict single next":
			self.X = tf.placeholder(tf.float32, [None, self.timelength, self.num_features])
			self.Y = tf.placeholder(tf.float32, [None, self.num_features])
			self.YClass = tf.placeholder(tf.int32, [None])
		elif self.batchType == "Predict <timestep> next" or self.batchType == "vanilla autoencoder":
			self.X = tf.placeholder(tf.float32, [None, self.timelength, self.num_features])
			self.Y = tf.placeholder(tf.float32, [None, self.timelength, self.num_features])
			self.YClass = tf.placeholder(tf.int32, [None])
		else:
			raise NotImplementedError()

	def createModel(self):
		# This creates the model including the necessary
		self.addPlaceholders()
		self.loss = None
		self.classificationLoss = None
		y_out = self.buildModel()
		self.loss, self.lossSummary, self.classificationLoss, self.classificationLossSummary = self.addLoss(y_out)
		if self.loss == None or self.classificationLoss == None:
			raise "You need to set and return both the loss and the classification loss in function addLoss"
		self.optimizer, self.classificationOptimizer = self.addOptimizer()
		if self.optimizer == None or self.classificationOptimizer == None:
			raise "You need to set and return both the optimizer and the classification optimizer in function addLoss"
		
		self.classificationAccuracy = None
		self.classificationAccuracy, self.classificationAccuracySummary = self.addAccuracy()
		if self.classificationAccuracy == None:
			raise "You need to set and return the classification accuracy in function addAccuracy"

		self.merged = tf.summary.merge([self.lossSummary])
		self.classificationMerged = tf.summary.merge([self.classificationAccuracySummary, self.classificationLossSummary])

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
		optimizer = tf.train.AdamOptimizer(self.lr_autoencoder)
		step1Train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"step1")
		optimizer = optimizer.minimize(self.loss, var_list=step1Train)
		optimizer2 = tf.train.AdamOptimizer(self.lr_classifier)
		step2Train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"step2")
		classificationOptimizer = optimizer2.minimize(self.classificationLoss, var_list=step2Train)
		return optimizer, classificationOptimizer

	def train(self):
		with tf.Session() as session:
			self.saver = tf.train.Saver()
			#logs_path = 'tensorboard/' + "_".join([str(x) for x in self.data.indices]) + '_' + strftime("%Y_%m_%d_%H_%M_%S", gmtime())
			logs_path = 'tensorboard/' + self.FLAGS.run_name
			train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)
			if self.FLAGS.load_dir == None:
				session.run(tf.global_variables_initializer())
			else:
				self.loadModel(session)

			if self.train_autoencoder:
				for i in range(self.iterations_autoencoder):
					#This is the part that trains the autoencoder
					batch_x, batch_y = self.data.getBatchOf(self.batch_size, self.timelength, self.batchType)
					summary, _ = session.run([self.merged, self.optimizer], feed_dict=self.createFeedDict(batch_x, batch_y))
					train_writer.add_summary(summary, i)
					if i % 300 == 0:
						#Every once in a while print the loss
						print ("Autoencoder Iteration: ", i)
						batch_x, batch_y = self.data.getBatchValid(self.batch_size, self.timelength, self.batchType)
						loss = session.run([self.loss], feed_dict=self.createFeedDict(batch_x, batch_y))
						print ("Loss = ", loss[0])
				if self.FLAGS.model_save_dir != None:
					self.saveModel(session, 'autoencoder/')

			if self.train_classifier:
				for i in range(self.iterations_classification):
					#This is the part that trains the classifier
					batch_x, batch_y = self.data.getBatchWithLabels(self.batch_size, self.timelength)
					session.run([self.classificationOptimizer], feed_dict=self.createFeedDict2(batch_x, batch_y))
					#train_writer.add_summary(summary, i)
					if i % 100 == 0:
						#Every once in a while print the loss
						print ("Classification Iteration: ", i)
						batch_x, batch_y = self.data.getBatchWithLabelsValid(self.batch_size, self.timelength)
						summary, loss, accuracy = session.run([self.classificationMerged, self.classificationLoss, self.classificationAccuracy], feed_dict=self.createFeedDict2(batch_x, batch_y))
						print ("Loss = ", loss)
						print ("Accuracy = ", accuracy)
				if self.FLAGS.model_save_dir != None:
					self.saveModel(session, 'autoencoder_and_classifier/')

			if self.compute_validation_accuracy:
				validationBatches = self.data.getAllValidationBatches(self.batch_size, self.timelength)
				# validationBatches = self.data.getAllValidationBatchesFromMiddle(self.batch_size, self.timelength)
				print ("Validation set number of batches =", len(validationBatches))
				accuracies = []
				for i, (batch_x, batch_y) in enumerate(validationBatches):
					print ("Validation iteration:", i)
					summary, loss, accuracy = session.run([self.classificationMerged, self.classificationLoss, self.classificationAccuracy], feed_dict=self.createFeedDict2(batch_x, batch_y))
					print ("Accuracy =", accuracy)
					accuracies.append(accuracy)
				accuracy = np.mean(accuracies)
				print accuracy 
				print ("Average accuracy on validation set =", accuracy) 

			if self.compute_test_accuracy:
				testBatches = self.data.getAllValidationBatches(self.batch_size, self.timelength)
				print ("Test set number of batches =", len(testBatches))
				accuracies = []
				for i, (batch_x, batch_y) in enumerate(testBatches):
					print ("Test iteration:", i)
					accuracy = session.run([self.classificationAccuracy], feed_dict=self.createFeedDict2(batch_x, batch_y))
					print ("Accuracy =", accuracy)
					accuracies.append(accuracy)
				accuracy = np.mean(accuracies)
				print accuracy 
				print ("Average accuracy on test set =", accuracy) 

	def saveModel(self, session, direc):
		model_save_path = self.FLAGS.model_save_dir
		directory = model_save_path + self.FLAGS.model_name
		if not os.path.exists(directory):
			os.makedirs(directory)
		directory += '/' + direc
		if not os.path.exists(directory):
			os.makedirs(directory)
		save_path = self.saver.save(session, directory + self.FLAGS.run_name)
		print("Model saved in file: %s" % save_path)

	def loadModel(self, session):
		print ('loading from: %s' %self.FLAGS.load_dir)
		self.saver.restore(session, self.FLAGS.load_dir)




