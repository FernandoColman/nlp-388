# models.py

from sentiment_data import *
from utils import *
from collections import Counter
from nltk.corpus import stopwords

import random
import numpy as np

stop_words = set(stopwords.words('english'))


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

    def preprocess_sentence(self, sentence: List[str]):
        cleaned_sentence = []
        for word in sentence:
            word = word.lower()
            if word not in stop_words:
                cleaned_sentence.append(word)
        return cleaned_sentence

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        # remove stopwords from sentence, lower all words
        # add to indexer if true
        c = Counter()
        self.preprocess_sentence(sentence)
        for word in sentence:
            if word:
                if add_to_indexer:
                    self.indexer.add_and_get_index(word)
                c[word] += 1
        return c


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def preprocess_sentence(self, sentence: List[str]):
        cleaned_sentence = []
        for word in sentence:
            word = word.lower()
            if word not in stop_words:
                cleaned_sentence.append(word)
        return cleaned_sentence

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        # remove stopwords from sentence, lower all words
        # add to indexer if true
        stopless_sentence = self.preprocess_sentence(sentence)

        c = Counter()
        ind = 0
        for i in range(len(stopless_sentence)):

            if ind + 1 < len(stopless_sentence):
                word_one = stopless_sentence[ind]
                word_two = stopless_sentence[ind + 1]
                if add_to_indexer:
                    self.indexer.add_and_get_index(word_one + word_two)
                c[word_one + word_two] += 1
                ind += 2
            elif ind + 1 == len(stopless_sentence):
                word_one = stopless_sentence[ind]
                if add_to_indexer:
                    self.indexer.add_and_get_index(word_one)
                c[word_one] += 1
            else:
                break
        return c


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    additional_words = ['.', ',', "'s", 'film', 'movie', '``', "''", "'", "`"]
    stop_words.update(additional_words)

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def preprocess_sentence(self, sentence: List[str]):
        cleaned_sentence = []
        for word in sentence:
            word = word.lower()
            if word not in stop_words:
                cleaned_sentence.append(word)
        return cleaned_sentence

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        c = Counter()
        self.preprocess_sentence(sentence)
        for word in sentence:
            if word:
                if add_to_indexer:
                    self.indexer.add_and_get_index(word)
                c[word] += 1
        return c


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
        feat_vector = np.zeros(shape=(len(self.weights[0]),))

        for word in sentence_counter:
            feat_vector[indexer.index_of(word)] = sentence_counter[word]

        pred = self.weights @ feat_vector

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
        sentence = self.feat_extractor.preprocess_sentence(sentence)
        sentence_counter = self.feat_extractor.extract_features(sentence)
        indexer = self.feat_extractor.get_indexer()
        feat_vector = np.zeros(shape=(len(self.weights[0]),))

        for word in sentence_counter:
            feat_vector[indexer.index_of(word)] = sentence_counter[word]

        pred = self.weights @ feat_vector

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
    weights = np.zeros(shape=(1, indexer.__len__()))

    alpha = .01
    epoch = 25
    perceptron = PerceptronClassifier(weights, feat_extractor)

    for epoch_count in range(1, epoch):
        random.shuffle(train_exs)

        for i in range(len(train_exs)):
            sentence = train_exs[i].words
            true_label = train_exs[i].label
            label_pred = perceptron.predict(sentence)

            # subtract when predicted 1 but actual 0
            if label_pred > true_label:
                for word in sentence:
                    if word:
                        feat_index = indexer.index_of(word)
                        weights[0][feat_index] -= alpha
                        if weights[0][feat_index] < -1:
                            weights[0][feat_index] = -1

            # add when predicted 0 but actual 1
            if label_pred < true_label:
                for word in sentence:
                    if word:
                        feat_index = indexer.index_of(word)
                        weights[0][feat_index] += alpha
                        if weights[0][feat_index] > 1:
                            weights[0][feat_index] = 1
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
    weights = np.zeros(shape=(1, indexer.__len__()))

    alpha = .01
    epoch = 50
    lr = LogisticRegressionClassifier(weights, feat_extractor)

    for epoch_count in range(1, epoch):
        random.shuffle(train_exs)

        for i in range(len(train_exs)):
            sentence = train_exs[i].words
            true_label = train_exs[i].label
            label_pred = lr.predict(sentence)
            gradient = sigmoid(label_pred - sigmoid(label_pred))

            # subtract when predicted 1 but actual 0
            if label_pred > true_label:
                for word in sentence:
                    if word:
                        feat_index = indexer.index_of(word)
                        weights[0][feat_index] -= alpha * gradient
                        if weights[0][feat_index] < -1:
                            weights[0][feat_index] = -1

            # add when predicted 0 but actual 1
            if label_pred < true_label:
                for word in sentence:
                    if word:
                        feat_index = indexer.index_of(word)
                        weights[0][feat_index] += alpha * gradient
                        if weights[0][feat_index] > 1:
                            weights[0][feat_index] = 1
    return lr


def preprocess(train_exs: List[SentimentExample]):
    random.shuffle(train_exs)
    cleaned_train_exs = []
    for i in range(len(train_exs)):
        for j in range(len(train_exs[i].words)):
            train_exs[i].words[j] = train_exs[i].words[j].lower()
            if train_exs[i].words[j] in stop_words:
                train_exs[i].words[j] = ""
        stopless_sentence = [x for x in train_exs[i].words if x]
        if stopless_sentence:
            cleaned_train_exs.append(SentimentExample(stopless_sentence, train_exs[i].label))
    return cleaned_train_exs


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
        train_exs = preprocess(train_exs)
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        train_exs = preprocess(train_exs)
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        train_exs = preprocess(train_exs)
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
