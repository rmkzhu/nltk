import pdb
import re
import nltk
import glob

def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = [i[0] for i in wordlist.most_common()]
    return word_features

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

def classify( classifier, tweet ):
    return classifier.classify(extract_features(tweet.split()))

training_size = 5000
pos_files = glob.glob("imdb/train/pos/*.txt")
neg_files = glob.glob("imdb/train/neg/*.txt")

positive = []
negative = []
counter = 0
for pos in pos_files:
  if counter >= training_size:
    break;
  with open(pos) as f:
    positive.append(re.sub(r'[^\s\w_]+',' ', f.read()))
  counter = counter + 1

counter = 0
for neg in neg_files:
  if counter >= training_size:
    break;
  with open(neg) as f:
    negative.append(re.sub(r'[^\s\w_]+',' ', f.read()))
  counter = counter + 1

training = []
for words in positive:
  words_filtered = [e.lower() for e in words.split() if len(e) >= 3] 
  training.append((words_filtered, 'positive'));
for words in negative:
  words_filtered = [e.lower() for e in words.split() if len(e) >= 3] 
  training.append((words_filtered, 'negative'));

print "Get word features"
word_features = get_word_features(get_words_in_tweets(training[:len(training)*4/5]))
print "Classify apply features"
training_set = nltk.classify.apply_features(extract_features, training[:len(training)*4/5])
word_features_test = get_word_features(get_words_in_tweets(training[len(training)*4/5:]))
test_set = nltk.classify.apply_features(extract_features, training[len(training)*4/5:])
print "Bayes train"
classifier = nltk.NaiveBayesClassifier.train( training_set )
print "Checking accuracy"
res = nltk.classify.util.accuracy(classifier, test_set)
print res

pdb.set_trace()
