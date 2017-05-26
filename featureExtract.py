#import librosa
import numpy as np
import random
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction

class dataHolder:
	def __init__(self):
		self.dirs = ['DC', 'JE', 'JK', 'KL']
		self.numbers = [15, 15, 15, 15, 30, 15, 15]
		self.fileTypes = ['a', 'd', 'f', 'h', 'n', 'sa', 'su']
		self.numbers = [15]
		self.fileTypes = ['a']
		filenames = self.getAllFilenames()
		self.DC, self.DCLabels = self.getAllFeatures('data/DC',filenames)
		print len(self.DC)
		self.DC, self.DCValid, self.DCtest, self.DCLabels, self.DCValidLabels, self.DCtestLabels = self.splitData(self.DC, self.DCLabels)
		self.JE, self.JELabels = self.getAllFeatures('data/JE',filenames)
		self.JE, self.JEValid, self.JEtest, self.JELabels, self.JEValidLabels, self.JEtestLabels = self.splitData(self.JE, self.JELabels)
		self.JK, self.JKLabels = self.getAllFeatures('data/JK',filenames)
		self.JK, self.JKValid, self.JKtest, self.JKLabels, self.JKValidLabels, self.JKtestLabels = self.splitData(self.JK, self.JKLabels)
		self.KL, self.KLLabels = self.getAllFeatures('data/KL',filenames)
		self.KL, self.KLValid, self.KLtest, self.KLLabels, self.KLValidLabels, self.KLtestLabels = self.splitData(self.KL, self.KLLabels)

	def splitData(self, data, dataLabels):
		valid = int(.8*float(len(data)))
		test = int(.9*float(len(data)))
		return data[0:valid], data[valid:test], data[test:], dataLabels[0:valid], dataLabels[valid:test], dataLabels[test:]

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
			returnLabels.append(item[endind])
		return np.array(returnBatch), np.array(returnLabels)

	def getBatchValid(self, size, length, varname="DC"):
		returnBatch = []
		returnLabels = []
		takenFrom = self.DCValid
		if varname == 'JE':
			takenFrom = self.JEValid
		elif varname == 'JK':
			takenFrom == self.JKValid
		elif varname == 'KL':
			takenFrom = self.KLValid
		for i in range(size):
			item = takenFrom[random.randint(0,len(takenFrom)-1)]
			endind = random.randint(length, len(item)-1)
			startind = endind-length
			returnBatch.append(item[startind:endind])
			returnLabels.append(item[endind])
		return np.array(returnBatch), np.array(returnLabels)

	def getBatchWithLabels(self, size, length, varname="DC"):
		returnBatch = []
		returnLabels = []
		labelsTaken = self.DCValidLabels
		takenFrom = self.DCValid
		if varname == 'JE':
			labelsTaken = self.JEValidLabels
			takenFrom = self.JEValid
		elif varname == 'JK':
			labelsTaken = self.JKValidLabels
			takenFrom == self.JKValid
		elif varname == 'KL':
			labelsTaken = self.KLValidLabels
			takenFrom = self.KLValid

		for i in range(size):
			index = random.randint(0,len(takenFrom)-1)
			item = takenFrom[index]
			endind = random.randint(length, len(item)-1)
			startind = endind-length
			returnBatch.append(item[startind:endind])
			returnLabels.append(labelsTaken[index])
		return np.array(returnBatch), np.array(returnLabels)

	def getBatchWithLabelsValid(self, size, length, varname="DC"):
		returnBatch = []
		returnLabels = []
		labelsTaken = self.DCLabels
		takenFrom = self.DC
		if varname == 'JE':
			labelsTaken = self.JELabels
			takenFrom = self.JE
		elif varname == 'JK':
			labelsTaken = self.JKLabels
			takenFrom == self.JK
		elif varname == 'KL':
			labelsTaken = self.KLLabels
			takenFrom = self.KL
			
		for i in range(size):
			index = random.randint(0,len(takenFrom)-1)
			item = takenFrom[index]
			endind = random.randint(length, len(item)-1)
			startind = endind-length
			returnBatch.append(item[startind:endind])
			returnLabels.append(labelsTaken[index])
		return np.array(returnBatch), np.array(returnLabels)

	def getAllFeatures(self, dirname, filenames):
		returnList = []
		returnLabels = []
		indices = [i for i in range(0,9)]
		indices.extend([i for i in range(22,34)])
		indices = [1]
		tot = np.zeros(len(indices))
		num = 0
		for el in filenames:
			s = dirname+'/'+el
			print (s)
			[Fs, x] = audioBasicIO.readAudioFile(s);
			F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);

			#a, _ = librosa.core.load(s)
			#features = librosa.feature.mfcc(a)
			#features = np.transpose(features)
			# features is a (time, features) array
			tot += np.mean(F[indices,:],axis=1)
			num +=1 
			returnList.append(F[indices,:].T)
			if el[0] == 'a':
				returnLabels.append(0)
			elif el[0] == 'd':
				returnLabels.append(1)
			elif el[0] == 'f':
				returnLabels.append(2)
			elif el[0] == 'h':
				returnLabels.append(3)
			elif el[0] == 'n':
				returnLabels.append(4)
			elif el[0:2] == 'sa':
				returnLabels.append(5)
			else:
				returnLabels.append(6)
		print (tot/num)
		return returnList, returnLabels


	def getAllFilenames(self):
		#fileTypes = ['a']
		#numbers = [5]
		returnList = []
		for i, el in enumerate(self.fileTypes):
			for j in range(self.numbers[i]):
				val = j+1
				s = el
				if val < 10:
					s = s + '0'
				s = s + str(val)
				s = s + '.wav'
				returnList.append(s)
		return returnList

