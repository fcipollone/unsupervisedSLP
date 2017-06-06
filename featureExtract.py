#import librosa
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import random
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from os import listdir, walk
import os
from os.path import isfile, join

class dataHolder:

	def __init__(self, FLAGS):

		self.featuresFileName = "features"
		self.labelsFileName = "labels"

		self.FLAGS = FLAGS
		self.indices = FLAGS.indices
		filenames = self.getAllFilenames()
		# self.testFileReading()
		self.data, self.labels = self.getAllFeatures(filenames)
		self.train, self.valid, self.test, self.train_labels, self.valid_labels, self.test_labels = self.splitData(self.data, self.labels)

	def splitData(self, data, dataLabels):
		valid = int(.8*float(len(data)))
		test = int(.9*float(len(data)))
		return data[0:valid], data[valid:test], data[test:], dataLabels[0:valid], dataLabels[valid:test], dataLabels[test:]

	def getBatchOf(self, size, length, out_type):
		returnBatch = []
		returnLabels = []
		takenFrom = self.train
		for i in range(size):
			item = takenFrom[random.randint(0,len(takenFrom)-1)]
			endind = random.randint(length, len(item)-1)
			startind = endind-length
			returnBatch.append(item[startind:endind])
			if out_type.lower() == "vanilla autoencoder":
				returnLabels.append(item[startind:endind])
			elif out_type.lower() == "predict single next":
				returnLabels.append(item[endind])
			elif out_type.lower() == "predict <timestep> next":
				returnLabels.append(item[startind+1:endind+1])
			else:
				raise NotImplementedError
		return np.array(returnBatch), np.array(returnLabels)

	def getBatchValid(self, size, length, out_type):
		returnBatch = []
		returnLabels = []
		takenFrom = self.valid
		for i in range(size):
			item = takenFrom[random.randint(0,len(takenFrom)-1)]
			endind = random.randint(length, len(item)-1)
			startind = endind-length
			returnBatch.append(item[startind:endind])
			if out_type.lower() == "vanilla autoencoder":
				returnLabels.append(item[startind:endind])
			elif out_type.lower() == "predict single next":
				returnLabels.append(item[endind])
			elif out_type.lower() == "predict <timestep> next":
				returnLabels.append(item[startind+1:endind+1])
			else:
				raise NotImplementedError
		return np.array(returnBatch), np.array(returnLabels)

	def getBatchWithLabels(self, size, length):
		returnBatch = []
		returnLabels = []
		takenFrom = self.train
		labelsTaken = self.train_labels
		for i in range(size):
			index = random.randint(0,len(takenFrom)-1)
			item = takenFrom[index]
			endind = random.randint(length, len(item)-1)
			startind = endind-length
			returnBatch.append(item[startind:endind])
			returnLabels.append(labelsTaken[index])
		return np.array(returnBatch), np.array(returnLabels)

	def getBatchWithLabelsValid(self, size, length):
		returnBatch = []
		returnLabels = []
		takenFrom = self.valid
		labelsTaken = self.valid_labels
		for i in range(size):
			index = random.randint(0,len(takenFrom)-1)
			item = takenFrom[index]
			endind = random.randint(length, len(item)-1)
			startind = endind-length
			returnBatch.append(item[startind:endind])
			returnLabels.append(labelsTaken[index])
		print np.shape(np.array(returnBatch)), np.shape(np.array(returnLabels))
		return np.array(returnBatch), np.array(returnLabels)

	def getAllValidationBatchesRandom(self, size, length):
		print "get all validation batches random"
		batches = []
		#random.seed(48392)
		batch_index = 0
		print "len(self.valid)", len(self.valid)
		print "size", size
		print "length", length
		while batch_index <= len(self.valid) - size:
			print batch_index
			batch = []
			labels = []
			for i in range(size):
				index = batch_index + i 
				item = self.valid[index]
				endind = random.randint(length, len(item)-1)
				startind = endind-length
				batch.append(item[startind:endind])
				labels.append(self.valid_labels[index])
				
			pairForBatch = [np.array(batch), np.array(labels)]
			batches.append(pairForBatch)
			batch_index += size 
		# print "shape", np.shape(batches)
		return batches

	def getAllValidationBatches(self, size, length):
		print "get all validation batches"
		batches = []
		batch_index = 0
		print "len(self.valid)", len(self.valid)
		print "size", size
		print "length", length

		# we want some number of batches
		# each batch is of length size 
		# each batch has a series of audio which are "length" timesteps

		# we enumerate through the files to create batches 
		batch = []
		labels = []
		for index, item in enumerate(self.valid):
			for startInd in [length*x for x in range(len(item)/length)]:
				if len(batch) == size:
					pairForBatch = [np.array(batch), np.array(labels)]
					batches.append(pairForBatch)
					batch = []
					labels = []
				endInd = startInd + length
				values = item[startInd:endInd]
				batch.append(values)
				labels.append(self.valid_labels[index])
		return batches

	def getAllValidationBatchesFromMiddle(self, size, length):
		print "get all validation batches"
		batches = []
		#random.seed(48392)
		batch_index = 0
		print "len(self.valid)", len(self.valid)
		print "size", size
		print "length", length

		# we want some number of batches
		# each batch is of length size 
		# each batch has a series of audio which are "length" timesteps

		# we enumerate through the files to create batches 
		batch = []
		labels = []
		for index, item in enumerate(self.valid):

			startIndices = [length*x for x in range(len(item)/length)]
			startIndicesMiddle = startIndices[len(startIndices)/3:len(startIndices) *2/3]

			for startInd in startIndicesMiddle:
				if len(batch) == size:
					pairForBatch = [np.array(batch), np.array(labels)]
					batches.append(pairForBatch)
					batch = []
					labels = []
				endInd = startInd + length
				values = item[startInd:endInd]
				batch.append(values)
				labels.append(self.valid_labels[index])
		return batches

	def getAllTestBatches(self, size, length):
		print "get all validation batches"
		batches = []
		batch_index = 0
		print "len(self.valid)", len(self.test)
		print "size", size
		print "length", length

		# we want some number of batches
		# each batch is of length size 
		# each batch has a series of audio which are "length" timesteps

		# we enumerate through the files to create batches 
		batch = []
		labels = []
		for index, item in enumerate(self.test):
			for startInd in [length*x for x in range(len(item)/length)]:
				if len(batch) == size:
					pairForBatch = [np.array(batch), np.array(labels)]
					batches.append(pairForBatch)
					batch = []
					labels = []
				endInd = startInd + length
				values = item[startInd:endInd]
				batch.append(values)
				labels.append(self.test_labels[index])
		return batches

	def getAllFeatures(self, filenames):
		allIndices = range(34)

		try:
   			features = np.load(self.featuresFileName + ".npy")
   			labels = np.load(self.labelsFileName + ".npy")
   			sliced_features = [x[:,self.indices] for x in features]
   			return sliced_features, labels

		except IOError:
			print "Generating features"
			features, labels = self.parseAllFeatures(allIndices, filenames)

			np.save(self.featuresFileName, features)
			np.save(self.labelsFileName, labels)

			sliced_features = [x[:,self.indices] for x in features]
			return sliced_features, labels

	def parseAllFeatures(self, indices, filenames):
		returnList = []
		returnLabels = []

		tot = np.zeros(len(indices))
		num = 0
		for el in filenames:
			classname = el.split('/')[-1].strip()
			print (el, classname)
			try:
				[Fs, x] = audioBasicIO.readAudioFile(el);
			except ValueError:
				continue
			F = None 
			if len(x.shape) == 1:
				F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);
			else:
				F = audioFeatureExtraction.stFeatureExtraction(x[:,0], Fs, 0.050*Fs, 0.025*Fs);

			tot += np.mean(F[indices,:],axis=1)
			num +=1 
			returnList.append(F[indices,:].T)
			if classname[0] == 'a':
				returnLabels.append(0)
			elif classname[0] == 'd':
				returnLabels.append(1)
			elif classname[0] == 'f':
				returnLabels.append(2)
			elif classname[0] == 'h':
				returnLabels.append(3)
			elif classname[0] == 'n':
				returnLabels.append(4)
			elif classname[0:2] == 'sa':
				returnLabels.append(5)
			else:
				returnLabels.append(6)
		returnListLength = len(returnList)

		random.seed(13921)

		shuffledIndices = random.sample(range(returnListLength), returnListLength)
		shuffledReturnList = [ returnList[i] for i in shuffledIndices]
		shuffledReturnLabels = [ returnLabels[i] for i in shuffledIndices]

		return shuffledReturnList, shuffledReturnLabels

		#return returnList, returnLabels


	def getAllFilenames(self):
		#fileTypes = ['a']
		#numbers = [5]
		returnList = []
		paths = []
		paths.extend(['RML2/s' + str(i+1) for i in range(8)])
		paths.extend(['data/DC', 'data/JE', 'data/JK', 'data/KL'])
		for mypath in paths:
			for (dirpath, dirnames, filenames) in walk(mypath):
				for f in filenames:
					if f.split('.')[1] == 'wav':
						returnList.append(dirpath + "/" + f)
		return returnList
