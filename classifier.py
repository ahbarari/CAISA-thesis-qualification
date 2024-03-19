import os
from collections import Counter
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

##########################
#  Feature computation
##########################

class FeatureComputer():
    def __init__(self,documents):
        self.docs = self.load_documents(documents)
        self.vocab = self.extract_vocabulary()
        self.idf = self.compute_idf(self.docs)
        self.vocab_index = self.get_vocab_index()

    def simple_features(self,document):
        """ Compute the simple features, i.e., number of sentences,
        the average number of words per sentence,
        and the average number of characters per word. """
        sentences = sent_tokenize(document)
        num_sent = len(sentences)
        mean_words = np.mean([len(word_tokenize(sent)) for sent in sentences])
        mean_chars = np.mean([len(word) for word in word_tokenize(document)])
        return num_sent, mean_words, mean_chars

    def load_documents(self,documents):
        """ Index and load documents """
        results = {}
        index = 0
        for doc, label in documents:
            results[index] = {'words': Counter(word_tokenize(doc)), 'label':label, 'doc':doc}
            index += 1
        return results

    def extract_vocabulary(self):
        """ Compute a dictionary indexing the vocabulary """
        vocab = {}
        for key, val in self.docs.items():
            for word in val['words']:
                if word in vocab:
                    vocab[word].add(key)
                else:
                    vocab[word] = {key}
        return vocab

    def get_vocab_index(self):
        """ Build vocabulary index dict """
        result = {word: i for i, word in enumerate(self.vocab)}
        return result

    def compute_idf(self, documents):
        """ Compute inverse document frequency dict for all words across
        all documents """
        results = {}
        total_docs = len(documents)
        for word, keys in self.vocab.items():
            results[word] = np.log(total_docs / len(keys))
        return results

    def compute_tf(self, word, document):
        """ Compute term frequency for the given word in a document """
        return document['words'].get(word) / len(document['words'])

    def compute_tf_idf(self, word, document):
        """ Compute TF-IDF """
        return self.compute_tf(word, document) * self.idf[word]
    
    def get_features_train(self):
        """ Compute training features for training data """
        examples = {}   
        for doc, document in sorted(self.docs.items()):
            feature = np.zeros(len(self.vocab_index))
            feature = np.append(feature, self.simple_features(document['doc']))
            for word, count in document['words'].items():
                if word in self.vocab_index:
                    feature[self.vocab_index[word]] = self.compute_tf_idf(word, document)
            examples[doc] = {'feature':feature, 'label':document['label']}
        return examples

    def get_features_test(self, testdata):
        """ Compute features for testing data """
        examples = {}
        test_docs = self.load_documents(testdata)
        for doc, document in sorted(test_docs.items()):
            feature = np.zeros(len(self.vocab_index))
            feature = np.append(feature, self.simple_features(document['doc']))
            for word, count in document['words'].items():
                if word in self.vocab_index:
                    feature[self.vocab_index[word]] = self.compute_tf_idf(word, document)
            examples[doc] = {'feature':feature, 
                             'label':document['label']}
        return examples

##########################
# Simple helper functions
##########################

def read_data(data):
    """ Parse the TSV file """
    result = []
    with open(data, 'r') as log:
        lines = log.readlines()
        for line in lines[1:]:  # Skip the first line, since it contains the description of the text columns
            data, label = line.strip().split('\t')
            result.append((data, label))
    return result

def get_best_features(x, y, remove_feature=None):
    """Computes the best feature by excluding one of the last three features at a time.
        remove_feature = None will only keep the TF-IDF vector
    """
    features = []

    for feature in x:
        current_feature_vector = feature[:-3]  # Remove the simple features
        if remove_feature is not None:
            # Remove the number of senteces
            if remove_feature == "num_sent":
                current_feature_vector = np.concatenate([feature[:-3], feature[-2:]])
            # Remove the average number of words per sentence
            elif remove_feature == "mean_words":
                current_feature_vector = np.concatenate([feature[:-2], feature[-1:]])
            # Remove the average number of characters per word
            elif remove_feature == "mean_chars":
                current_feature_vector = feature[:-1]
            # Remove the average number of words per sentence and the average number of characters per word
            elif remove_feature == "all_mean":
                current_feature_vector = feature[:-2]
        features.append(current_feature_vector)

    features = np.array(features)

    return features, y

##########################
#       Classifier
##########################

path = os.getcwd()
random_seed = 42

print("Loading data...")

train = read_data('data/train.tsv')
test = read_data('data/test.tsv')

print ("Computing features...")

feature_comp = FeatureComputer(train)
data_train = feature_comp.get_features_train()
data_test = feature_comp.get_features_test(test)

# Imputer for missing values in the test data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit([doc['feature'] for key,doc in data_train.items()])

# Get feature vector and label for training
train_X = [doc['feature'] for key,doc in sorted(data_train.items())]
train_y = [doc['label'] for key, doc in sorted(data_train.items())]

# Get feature vector and label for testing
test_X = imputer.transform([doc['feature'] for key,doc in sorted(data_test.items())])
test_y = [doc['label'] for key, doc in sorted(data_test.items())]

print ("Training models...")

# Initialize and Train Logistic Regression model
logistic_model = LogisticRegression(random_state=random_seed)
logistic_model.fit(train_X, train_y)

# Initialize and Train MLP model
mlp_model = MLPClassifier(random_state=random_seed)
mlp_model.fit(train_X, train_y)

print("Computing scores for all features...")

# Model Predection on test data
logistic_predictions = logistic_model.predict(test_X)
mlp_predictions = mlp_model.predict(test_X)


logistic_accuracy = accuracy_score(test_y, logistic_predictions)
mlp_accuracy = accuracy_score(test_y, mlp_predictions)


print("Logistic Regression Accuracy: {:.2f}".format(logistic_accuracy))
print("MLP Accuracy: {:.2f}".format(mlp_accuracy))


logistic_repot = classification_report(test_y, logistic_predictions)

print(logistic_repot)

mlp_report = classification_report(test_y, mlp_predictions)

print(mlp_report)


print("-------------------------------------------------------------")


best_train_X, best_train_y = get_best_features(train_X, train_y, "num_sent")
best_test_X, best_test_y = get_best_features(test_X, test_y, "num_sent")


best_model_logistic = LogisticRegression(random_state=random_seed)
best_model_logistic.fit(best_train_X, best_train_y)

best_model_mlp = MLPClassifier(random_state=random_seed)
best_model_mlp.fit(best_train_X, best_train_y)

print("Computing scores based on modified features...")

best_logistic_predictions = best_model_logistic.predict(best_test_X)
best_mlp_predictions = best_model_mlp.predict(best_test_X)


best_logistic_accuracy = accuracy_score(best_test_y, best_logistic_predictions)
best_mlp_accuracy = accuracy_score(test_y, best_mlp_predictions)


print("Logistic Regression with Modified Features Accuracy: {:.2f}".format(best_logistic_accuracy))
print("MLP with Modified Features Accuracy: {:.2f}".format(best_mlp_accuracy))


best_logistic_repot = classification_report(best_test_y, best_logistic_predictions)

print(best_logistic_repot)

best_mlp_report = classification_report(best_test_y, best_mlp_predictions)

print(best_mlp_report)