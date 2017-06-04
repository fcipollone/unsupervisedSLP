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
	def __init__(self,data):

		self.indices = [6]
		filenames = self.getAllFilenames()
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
			elif out_type.lower() == "predict <timesteps> next":
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
			elif out_type.lower() == "predict <timesteps> next":
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
		return np.array(returnBatch), np.array(returnLabels)

	def getAllFeatures(self, filenames):
		returnList = []
		returnLabels = []

		indices = self.indices

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

		shuffledIndices = random.sample(range(returnListLength), returnListLength)
		shuffledReturnList = [ returnList[i] for i in shuffledIndices]
		shuffledReturnLabels = [ returnLabels[i] for i in shuffledIndices]

		return shuffledReturnList, shuffledReturnLabels


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
