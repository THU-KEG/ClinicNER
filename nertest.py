from itertools import chain
import os
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
import i2b2k_data
import numpy as np
import data_utils
print(sklearn.__version__)

train_sents = i2b2k_data.iob_sents('train')
test_sents = i2b2k_data.iob_sents('test')

base_dir = 'i2b2k/2010'
embedding_files = ['GoogleNews-vectors-negative300.bin.emb.npy','glove.840B.300d.txt.emb.npy']
embedding_file_name = embedding_files[1]
embedding_path = os.path.join(base_dir, embedding_file_name)
vocab_path = os.path.join(base_dir,'vocab.txt')

word2id,_ = data_utils.initialize_vocabulary(vocab_path)
embedding = np.load(embedding_path)
#print word2id
#print 'embedding',embedding[word2id['the']]

print 'calculate cagegory center....'
category2mentions = data_utils.initialize_category(os.path.join(base_dir,'category.txt'))
category2center = data_utils.caculate_category_center(category2mentions, word2id, embedding)
#print category2center


def cateogry_center_features(category2center,word):
    if word in word2id:
        features = {}
        vec = embedding[word2id[word]]
        for k in category2center:
            euclidean_distance = np.linalg.norm(vec - category2center[k])
            cosine_sim = _cal_consine_sim(vec, category2center[k])
            features['%sCenterDis'% k] = euclidean_distance
            features['%sCosineSim'% k] = cosine_sim
        return features
    else:
        raise ValueError("Vocabulary %s not found.", word)

def _cal_consine_sim(vec1,vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return np.dot(vec1, vec2)/(norm1*norm2)

def word2vec_features(word):
    if word in word2id:
        features = {}
        vec = embedding[word2id[word]]
        for i in xrange(0,len(vec)):
            features['vec%d'%i] = float(vec[i])
        return features
    else:
        raise ValueError("Vocabulary %s not found.", word)
def _word_shape_transfer(word):
    buf = []
    for i in word:
        i = i.encode('utf-8')
        if not str.isalpha(i):
            buf.append(i)
        else:
            if str.isupper(i):
                buf.append('X')
            else:
                buf.append('x')
    return ''.join(buf)
def word2features(sent, i):
    word = sent[i][0]
    wordshape = _word_shape_transfer(word)
    postag = sent[i][1]
    ###basic features
    features = {
        'bias':1.0,
        'word.lower=' + word.lower():1.0,
        'word[-4:]=' + word[-4:]:1.0,
        'word[-3:]=' + word[-3:]:1.0,
        'word[-2:]=' + word[-2:]:1.0,
        'word[0:1]=' + word[0:1]:1.0,
        'word[0:2]=' + word[0:2]:1.0,
        'word[0:3]=' + word[0:3]:1.0,
        'wordshaple=%s' % wordshape:1.0,
        'word.isupper=%s' % word.isupper():1.0,
        'word.istitle=%s' % word.istitle():1.0,
        'word.isdigit=%s' % word.isdigit():1.0,
        'postag=' + postag:1.0,
        'postag[:2]=' + postag[:2]:1.0
    }
    if i > 0:
        word1 = sent[i-1][0]
        word1shape = _word_shape_transfer(word1)
        postag1 = sent[i-1][1]
        newfeatures = {
            '-1:word.lower=' + word1.lower():1.0,
            '-1:word.istitle=%s' % word1.istitle():1.0,
            '-1:word.isupper=%s' % word1.isupper():1.0,
            '-10:word.lower=' + word1.lower()+'_'+word.lower():1.0,
            '-1:postag=' + postag1:1.0,
            '-1:isdigit=%s' % word.isdigit():1.0,
            '-1:postag[:2]=' + postag1[:2]:1.0
        }
        features = dict(features,**newfeatures)
    else:
        features['BOS'] = 1.0
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        word1shape = _word_shape_transfer(word1)
        postag1 = sent[i+1][1]
        newfeatures = {
            '+1:word.lower=' + word1.lower():1.0,
            '+1:word.istitle=%s' % word1.istitle():1.0,
            '+1:word.isupper=%s' % word1.isupper():1.0,
            '01:word.lower=' +word.lower()+'_'+word1.lower():1.0,
            '+1:isdigit=%s' % word1.isdigit():1.0,
            '+1:postag=' + postag1:1.0,
            '+1:postag[:2]=' + postag1[:2]:1.0
        }
        features = dict(features,**newfeatures)
    else:
        features['EOS'] = 1.0
    ### word2vec features
    newfeatures = word2vec_features(word.lower())
    features = dict(features,**newfeatures)

    ### cateogry features
    newfeatures = cateogry_center_features(category2center,word.lower())
    features = dict(features,**newfeatures)
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]
#print X_train[0]
#print y_train[0]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]


trainer = pycrfsuite.Trainer(verbose=True)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
    'c1': 0.5,   # coefficient for L1 penalty
    'c2': 1e-4,  # coefficient for L2 penalty
    'max_iterations': 100,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})
trainer.train('i2b2k2010-esp.crfsuite')



##############################test################
tagger = pycrfsuite.Tagger()
tagger.open('i2b2k2010-esp.crfsuite')
example_sent = test_sents[0]
print ' '.join(sent2tokens(example_sent))+'\n\n'

print "Predicted:", ' '.join(tagger.tag(sent2features(example_sent)))
print "Correct:  ", ' '.join(sent2labels(example_sent))

def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )
y_pred = [tagger.tag(xseq) for xseq in X_test]
print y_pred[0],y_test[0],test_sents[0]
#output result
assert len(y_pred) == len(test_sents)
with open(os.path.join(base_dir,'result.txt'),'w') as f:
    buflines =[]
    for no in xrange(0,len(y_pred)):
        buflines.extend(['\t'.join(list(i[0])+[i[1]])+'\n' for i in zip(test_sents[no],y_pred[no])])
        buflines.extend(['\n'])
    f.writelines(buflines)
print(bio_classification_report(y_test, y_pred))