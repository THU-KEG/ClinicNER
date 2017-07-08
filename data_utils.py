import os
import re
from codecs import open as codecs_open
import cPickle as pickle
import numpy as np

base_dir = 'i2b2k/2010'

#load txt vocab
def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.
    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/data_utils.py
    """
    if os.path.exists(vocabulary_path):
        rev_vocab = []
        with codecs_open(vocabulary_path, "rb", encoding="utf-8") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def _load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    Original taken from
    https://github.com/yuhaozhang/sentence-convnet/blob/master/text_input.py
    """
    print 'load bin...'
    word_vecs = {}
    cnt = 0

    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        print 'embedding vocab size: %d , embedding vector size: %d'%(vocab_size,layer1_size)
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
            cnt +=1
            if cnt%10000 == 0:
            	print '%d lines...'%cnt
    return (word_vecs, layer1_size)
def _load_glove_vec(fname, vocab):
    """
    load word  vectors from glove model using glove-python.
    dependency: https://github.com/maciejkula/glove-python
    """
    print 'load glove...'
    word_vecs = {}
    cnt = 0
    l = open(fname,'r').readline()
    embedding_size = len(l.strip().split()) -1
    print 'embedding vector size: %d'%(embedding_size)
    with open(fname, "r") as f:
        for l in f:
            stemp = l.strip().split(' ',1)
            assert len(stemp) == 2
            word = stemp[0]
            if word in vocab:
                word_vecs[stemp[0]] = np.fromstring(' '.join(stemp[1:]),sep = ' ')
            cnt+=1
            if cnt%10000==0:
                print '%d lines...'%cnt
    return (word_vecs,embedding_size)

def _add_random_vec(word_vecs, vocab, emb_size=300):
	print 'add random...'
	allcnt = 0
	unkcnt = 0
	for word in vocab:
		allcnt+=1
		if word not in word_vecs:
			unkcnt+=1
			word_vecs[word] = np.random.uniform(-0.25,0.25,emb_size)
	print 'all vocab size:%d  unk vocab in word2vec:%d'%(allcnt,unkcnt)
	return word_vecs


def prepare_pretrained_embedding(fname, word2id,load_method=_load_bin_vec):
    print 'Reading pretrained word vectors from file ...'
    word_vecs, emb_size = load_method(fname, word2id)
    word_vecs = _add_random_vec(word_vecs, word2id, emb_size)
    embedding = np.zeros([len(word2id), emb_size])
    for w,idx in word2id.iteritems():
        embedding[idx,:] = word_vecs[w]
    print 'Generated embeddings with shape ' + str(embedding.shape)
    return embedding


def initialize_category(fname):
    category2mentions ={}
    with open(fname,'r') as f:
        for line in f:
            stemp = line.split('\t')
            category2mentions[stemp[0].strip()] = [i.strip() for i in stemp[1].split(':::')]
            assert len(category2mentions[stemp[0].strip()]) > 0
    return category2mentions


def caculate_category_center(category2mentions,word2id,embeddings):
    assert word2id != None 
    category2center = {}
    vec_size = len(embeddings[0])
    for k in category2mentions.keys():
        cnt = 0
        err = 0
        vec = np.zeros(vec_size)
        for m in category2mentions[k]:
            m_vec = _caculate_mention_vec(m,word2id,embeddings)
            if m_vec is None:
                err+=1
            else:
                vec = vec + m_vec
                cnt +=1
        vec/=cnt
        category2center[k] = vec
        print 'category:%s  cnt:%d  err:%d' % (k,cnt,err)
    return category2center


def _caculate_mention_vec(mention,word2id,embeddings):
    vec_size = len(embeddings[0])
    mention_vec = np.zeros(vec_size)
    cnt  = 0
    for w in mention.split():
        cnt +=1
        w = w.lower().strip()
        if w in word2id:
            mention_vec+=embeddings[word2id[w]]
        else:
            return None
            #raise ValueError("Vocabulary %s not found. in mention %s" % (w,mention))
    mention_vec/=cnt
    return mention_vec


def main():
    embedding_files = ['GoogleNews-vectors-negative300.bin','glove.840B.300d.txt','mimic_glove.txt']
    vocab_path = os.path.join(base_dir,'vocab.txt')
    embedding_file_name = embedding_files[2]
    load_method = _load_glove_vec
    embedding_path = os.path.join('word2vec', embedding_file_name)
    if os.path.exists(embedding_path):
        word2id, _ = initialize_vocabulary(vocab_path)
        embedding = prepare_pretrained_embedding(embedding_path, word2id,load_method)
        np.save(os.path.join(base_dir, embedding_file_name+'.emb.npy'), embedding)
    else:
        print "Pretrained embeddings file %s not found." % embedding_path
if __name__ == '__main__':
    main()