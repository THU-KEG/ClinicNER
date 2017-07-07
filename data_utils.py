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
            	print word
                f.read(binary_len)
            cnt +=1
            if cnt%10000 == 0:
            	print '%d lines...'%cnt
    return (word_vecs, layer1_size)


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


def prepare_pretrained_embedding(fname, word2id):
    print 'Reading pretrained word vectors from file ...'
    word_vecs, emb_size = _load_bin_vec(fname, word2id)
    word_vecs = _add_random_vec(word_vecs, word2id, emb_size)
    embedding = np.zeros([len(word2id), emb_size])
    for w,idx in word2id.iteritems():
        embedding[idx,:] = word_vecs[w]
    print 'Generated embeddings with shape ' + str(embedding.shape)
    return embedding


def initialize_category(fname):
    category2mentions

def main():
    vocab_path = os.path.join(base_dir,'vocab.txt')
    embedding_file_name = 'GoogleNews-vectors-negative300.bin'
    embedding_path = os.path.join('word2vec', embedding_file_name)
    if os.path.exists(embedding_path):
        word2id, _ = initialize_vocabulary(vocab_path)
        embedding = prepare_pretrained_embedding(embedding_path, word2id)
        np.save(os.path.join(base_dir, embedding_file_name+'.emb.npy'), embedding)
    else:
        print "Pretrained embeddings file %s not found." % embedding_path
if __name__ == '__main__':
    main()