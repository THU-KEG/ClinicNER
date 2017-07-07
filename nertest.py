from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
import i2b2k_data

print(sklearn.__version__)
#print nltk.corpus.conll2002.fileids()
#train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
train_sents = i2b2k_data.iob_sents('train')
#test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
test_sents = i2b2k_data.iob_sents('test')
print train_sents[0]

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'bias':1.0,
        'word.lower=' + word.lower():1.0,
        'word[-3:]=' + word[-3:]:1.0,
        'word[-2:]=' + word[-2:]:1.0,
        'word.isupper=%s' % word.isupper():1.0,
        'word.istitle=%s' % word.istitle():1.0,
        'word.isdigit=%s' % word.isdigit():1.0,
        'postag=' + postag:1.0,
        'postag[:2]=' + postag[:2]:1.0
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        newfeatures = {
            '-1:word.lower=' + word1.lower():1.0,
            '-1:word.istitle=%s' % word1.istitle():1.0,
            '-1:word.isupper=%s' % word1.isupper():1.0,
            '-10:word.lower=' + word1.lower()+'_'+word.lower():1.0,
            '-1:postag=' + postag1:1.0,
            '-1:postag[:2]=' + postag1[:2]:1.0
        }
        features = dict(features,**newfeatures)
    else:
        features['BOS'] = 1.0
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        newfeatures = {
            '+1:word.lower=' + word1.lower():1.0,
            '+1:word.istitle=%s' % word1.istitle():1.0,
            '+1:word.isupper=%s' % word1.isupper():1.0,
            '01:word.lower=' +word.lower()+'_'+word1.lower():1.0,
            '+1:postag=' + postag1:1.0,
            '+1:postag[:2]=' + postag1[:2]:1.0
        }
        features = dict(features,**newfeatures)
    else:
        features['EOS'] = 1.0
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]
print X_train[0]
print y_train[0]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]


trainer = pycrfsuite.Trainer(verbose=True)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

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
print(bio_classification_report(y_test, y_pred))