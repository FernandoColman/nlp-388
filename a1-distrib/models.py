# models.py

from sentiment_data import *
from utils import *
from collections import Counter
from nltk.corpus import stopwords

import random
import numpy as np

stop_words = set(stopwords.words('english'))
additional_words = ['.', ',', "'s", 'film', 'movie']
stop_words.update(additional_words)


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        # remove stopwords from sentence, lower all words
        # add to indexer if true
        c = Counter()
        for word in sentence:
            word = word.lower()
            if word not in stop_words:
                if add_to_indexer:
                    self.indexer.add_and_get_index(word)
                c[word] += 1
        return c


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """

    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, weights, feat_extractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        sentence_counter = self.feat_extractor.extract_features(sentence)
        indexer = self.feat_extractor.get_indexer()
        feat_vector = np.zeros(len(self.weights))

        for word in sentence_counter:
            feat_vector[indexer.index_of(word)] = 1

        pred = np.dot(self.weights, feat_vector)

        if pred > 0:
            return 1
        return 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, weights, feat_extractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> float:
        sentence_counter = self.feat_extractor.extract_features(sentence)
        indexer = self.feat_extractor.get_indexer()
        feat_vector = np.zeros(len(self.weights))

        for word in sentence_counter:
            feat_vector[indexer.index_of(word)] = 1

        pred = np.dot(self.weights, feat_vector)

        if pred > 0:
            return 1
        return 0


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """

    # count all features and update index
    for sentence in train_exs:
        feat_extractor.extract_features(sentence.words, True)

    # create weight vector based off index size
    indexer = feat_extractor.get_indexer()
    weights = np.zeros(indexer.__len__())

    alpha = .5
    epoch = 10
    perceptron = PerceptronClassifier(weights, feat_extractor)

    for epoch_count in range(1, epoch):
        random.shuffle(train_exs)

        if epoch_count % 2 == 0:
            alpha -= .01
        for i in range(len(train_exs)):
            label_pred = perceptron.predict(train_exs[i].words)

            # subtract when predicted 1 but actual 0
            if label_pred > train_exs[i].label:
                for word in train_exs[i].words:
                    weights[indexer.index_of(word)] -= alpha
                    if weights[indexer.index_of(word)] < -1:
                        weights[indexer.index_of(word)] = -1

            # add when predicted 0 but actual 1
            if label_pred < train_exs[i].label:
                for word in train_exs[i].words:
                    weights[indexer.index_of(word)] += alpha
                    if weights[indexer.index_of(word)] > 1:
                        weights[indexer.index_of(word)] = 1
    return perceptron


def train_logistic_regression(train_exs: List[SentimentExample],
                              feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # count all features and update index
    for sentence in train_exs:
        feat_extractor.extract_features(sentence.words, True)

    # create weight vector based off index size
    indexer = feat_extractor.get_indexer()
    weights = np.zeros(indexer.__len__())

    alpha = 1
    epoch = 10
    lr = LogisticRegressionClassifier(weights, feat_extractor)

    for epoch_count in range(1, epoch):
        random.shuffle(train_exs)
        if epoch_count % 2 == 0:
            alpha -= .01
        for i in range(len(train_exs)):
            label_pred = lr.predict(train_exs[i].words)

            # subtract when predicted 1 but actual 0
            if label_pred > train_exs[i].label:
                for word in train_exs[i].words:
                    weights[indexer.index_of(word)] -= alpha * (1 - sigmoid(label_pred))
                    if weights[indexer.index_of(word)] < -1:
                        weights[indexer.index_of(word)] = -1

            # add when predicted 0 but actual 1
            if label_pred < train_exs[i].label:
                for word in train_exs[i].words:
                    weights[indexer.index_of(word)] += alpha * (1 - sigmoid(label_pred))
                    if weights[indexer.index_of(word)] > 1:
                        weights[indexer.index_of(word)] = 1

    return lr


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        random.shuffle(train_exs)
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model
