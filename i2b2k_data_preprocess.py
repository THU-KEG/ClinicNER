import os
import re
import nltk
import sys
base_dir = 'i2b2k/2010'
train_dir = os.path.join(base_dir,'concept_assertion_relation_training_data/beth/')
train_labeldir = os.path.join(train_dir,'concept')
train_txtdir = os.path.join(train_dir,'txt')
test_dir = os.path.join(base_dir,'reference_standard_for_test_data/')
test_labeldir = os.path.join(test_dir,'concepts')
test_txtdir = os.path.join(base_dir,'test_data/')

train_outputpath = os.path.join(base_dir,'train.txt')
test_outputpath = os.path.join(base_dir,'test.txt')
category_outputpath = os.path.join(base_dir,'category.txt')

category2mentions = {}
error = 0

#preprocess data and output in CONLL BIO format
def rawdata2bio(text_dir, label_dir, outputpath):
	print text_dir
	writebuf = []
	for p,d,fs in os.walk(text_dir):
		for f in fs:
			if f[-3:] != 'txt':
				continue
			txtpath = os.path.join(p,f)
			labelpath = os.path.join(label_dir,f)
			labelpath = labelpath[:-3]+'con'
			textlines = open(txtpath,'r').readlines()
			labellines = open(labelpath,'r').readlines()
			generateBIO(textlines,labellines,writebuf,tokenizer = naiveTokenize)
	out = open(outputpath,'w')
	out.writelines(writebuf)
def naiveTokenize(sent):
	tokens = sent.split()
	tagged = nltk.pos_tag(tokens)
	#print tagged
	return [[i[0],i[1],'O'] for i in tagged]

def generateBIO(textlines,labellines,writebuf,tokenizer = naiveTokenize):
	global error
	tokens = [tokenizer(line.strip()) for line in textlines]
	#print tokens
	for line in labellines:
		if len(line) == 0:
			continue
		m = re.search('c=\"(.*?)\"\s(\d+):(\d+)\s(\d+):(\d+)\|\|t=\"(.*?)\"', line)
		assert m
		infos = [m.group(i) for i in range(1,7)]
		mention = infos[0]
		##mention maybe contain more than 1 blank
		mention = ' '.join(mention.split())
		lineno = int(infos[1])-1
		bindex = int(infos[2])
		eindex = int(infos[4])
		category = infos[5]
		#print '\t'.join([m.group(i) for i in range(1,7)])
		#print textlines[lineno].strip()
		category2mentions.setdefault(category, set()).add(mention)
		tokens_mention = ' '.join([tokens[lineno][i][0] for i in range(bindex,eindex+1)]).lower()
		#print tokens[lineno][bindex],mention,tokens_mention
		if tokens_mention !=mention:
			print '%s not equal %s'%(tokens_mention,mention)
			error+=1
			#sys.exit(-1)
		###modify BIO
		for i in range(bindex,eindex+1):
			if tokens[lineno][i][2]!= 'O':
				break
			tokens[lineno][i][2] = 'B-'+category.upper() if i == bindex else 'I-'+category.upper()
	for line in tokens:
		###filt all 'O' sentences
		allO = True
		for token in line:
			if token[2] != 'O':
				allO = False
				break
		if not allO:
			writebuf.extend(t[0]+'\t'+t[1]+'\t'+t[2]+'\n' for t in line)
			writebuf.append('\n')


def main():
	rawdata2bio(train_txtdir,train_labeldir, train_outputpath)
	rawdata2bio(test_txtdir,test_labeldir, test_outputpath)


	train_dir = os.path.join(base_dir,'concept_assertion_relation_training_data/partners/')
	train_labeldir = os.path.join(train_dir,'concept')
	train_txtdir = os.path.join(train_dir,'txt')
	rawdata2bio(train_txtdir,train_labeldir, train_outputpath+'.partners.txt')

	#print category2mentions,len(category2mentions)
	#print category2mentions.keys()
	print 'error number:%s'%error

	out = open(category_outputpath,'w')
	for key in category2mentions:
		out.write(key+'\t'+':::'.join([i for i in category2mentions[key]])+'\n')

if __name__=='__main__':
	main()

