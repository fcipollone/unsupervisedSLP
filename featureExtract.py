import librosa
import numpy as np
import random

class dataHolder:
	def __init__(self):
		self.dirs = ['DC', 'JE', 'JK', 'KL']
		filenames = self.getAllFilenames()
		self.DC = self.getAllFeaures('data/DC',filenames)
		self.JE = self.getAllFeaures('data/JE',filenames)
		self.JK = self.getAllFeaures('data/JK',filenames)
		self.KL = self.getAllFeaures('data/KL',filenames)

	def getBatchOf(self, size, length, varname="DC"):
		returnBatch = []
		returnLabels = []
		takenFrom = self.DC
		if varname == 'JE':
			takenFrom = self.JE
		elif varname == 'JK':
			takenFrom == self.JK
		elif varname == 'KL':
			takenFrom = self.KL
		for i in range(size):
			item = takenFrom[random.randint(0,len(takenFrom)-1)]
			endind = random.randint(length, len(item)-1)
			startind = endind-length
			returnBatch.append(item[startind:endind])
			returnLabels.append(item[startind+1:endind+1])
		return np.array(returnBatch), np.array(returnLabels)


	def getAllFeaures(self, dirname, filenames):
		returnList = []
		for el in filenames:
			s = dirname+'/'+el
			print (s)
			a, _ = librosa.core.load(s)
			features = librosa.feature.mfcc(a)
			features = np.transpose(features)
			# features is a (time, features) array
			returnList.append(features)

		return returnList


	def getAllFilenames(self):
		fileTypes = ['a', 'd', 'f', 'h', 'n', 'sa', 'su']
		numbers = [15, 15, 15, 15, 30, 15, 15]
		#fileTypes = ['a']
		#numbers = [5]
		returnList = []
		for i, el in enumerate(fileTypes):
			for j in range(numbers[i]):
				val = j+1
				s = el
				if val < 10:
					s = s + '0'
				s = s + str(val)
				s = s + '.wav'
				returnList.append(s)
		return returnList

