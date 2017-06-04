from os import walk
import subprocess

returnList = []
paths = []
paths.extend(['s' + str(i+1) for i in range(8)])
for mypath in paths:
	for (dirpath, dirnames, filenames) in walk(mypath):
		for f in filenames:
			if f.split('.')[1] == 'avi':
				returnList.append(dirpath + "/" + f)
				print dirpath + "/" + f
				print  dirpath + '/' + f.split('.')[0]
				command = "ffmpeg -i " + dirpath + "/" + f + " -ab 160k -ac 2 -ar 44100 -vn " + dirpath + '/' + f.split('.')[0] + ".wav"
				subprocess.call(command, shell=True)

#print returnList