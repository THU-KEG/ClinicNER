import os

base_dir = 'i2b2k/2010'

def iob_sents(dataset):
	path = os.path.join(base_dir,'%s.txt'%dataset) 
	assert os.path.exists(path)
	buf = []
	res = []
	fin = open(path,'r')
	line = fin.readline()
	while line:
		if line:
			if line.strip() == '' and len(buf)>0:
				res.append([i for i in buf])
				buf = []
			else:
				stemp = line.split()
				buf.append(tuple(map(unicode,stemp)))
			line = fin.readline()
		else:
			return res
	return res

sents = iob_sents('train')
print sents[0]
print sents[-1]