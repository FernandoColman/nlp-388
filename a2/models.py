# models.py

import torch
import torch.nn as nn
from numpy import mean
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)

    in intialize function set up model, indexer
    predict function based on embeddings
        set model to evaluation mode, don't learn from indexes. if not it'll use your own embeddings and not GloVe
    """

    def __init__(self, word_embeddings, hidden_dim, num_classes, has_typos: bool):
        self.has_typos = has_typos
        self.word_embeddings = word_embeddings
        self.model = DAN(word_embeddings, hidden_dim, num_classes)

    def predict(self, sentence: List[str], has_typos: bool) -> int:
        self.model.eval()
        with torch.no_grad():
            probs = self.model(sentence)
            return torch.argmax(probs).item()



class DAN(nn.Module):
    """
    Constructs the computation graph by instantiating the various layers and initializing weights.

    :param inp: size of input (integer) -> all the sentences
    :param hid: size of hidden layer(integer)
    :param out: size of output (integer) -> e the number of classes

    sub implementaiton of nn.Module
        take average of embeddings and THEN use forward() to every embeddings, then apply that average to the model's
        output. all of this happening in init and forward
    """

    def __init__(self, word_embeddings: WordEmbeddings, hidden_dim, num_classes):
        super(DAN, self).__init__()
        self.embeddings = word_embeddings
        self.embed_layer = word_embeddings.get_initialized_embedding_layer()

        self.W1 = nn.Linear(self.embed_layer.embedding_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, num_classes)
        self.g = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=0)

        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param word_indices: a [inp]-sized tensor of input data -> word indices
            1. replace words with their embeddings
            2. reduce each sentence to one dimension, using the average of all the word embeddings
        :return: an [out]-sized tensor of log probabilitires.
        """

        # finds the embeddings for each word. one word will have 50 or 300 embeddings
        word_indices = self.get_index_layer(x)
        embeds_for_words = self.embed_layer(torch.tensor(word_indices))

        # averages the embeddings of each word, now each embedding is averaged from all the words
        avg_embeds = torch.mean(embeds_for_words, dim=0)

        hidden_layer = self.W1(avg_embeds)
        non_linearity = self.g(hidden_layer)
        output_layer = self.W2(non_linearity)
        probs = self.log_softmax(output_layer)  # range: [-inf, 0], closer to 0 means higher prob
        return probs

    def get_index_layer(self, sentence) -> list:
        # grab words if sentence is a SentimentExample
        if isinstance(sentence, SentimentExample):
            sentence = sentence.words

        index_layer = []
        for word in sentence:
            word_index = self.embeddings.word_indexer.index_of(word)
            if word_index > -1:
                index_layer.append(word_index)
        return index_layer


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings,
                                 train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.

    get embeddings, define epocs learning rate wtv, instantiate model and set up for loss (neg log likelihood)
    then implement neural sentiment classifier, for every example:
        - word index layer
            also send unknown word to UNK
            make sure to handle unknown indexers
        - set optimizer to 0 grad
        - model impl
        -
    """

    epochs = 10
    learning_rate = 0.001
    hidden_dim = 10
    num_classes = 2
    loss_crit = nn.CrossEntropyLoss()

    nsc = NeuralSentimentClassifier(word_embeddings, hidden_dim, num_classes, train_model_for_typo_setting)
    optimizer = optim.Adam(nsc.model.parameters(), lr=learning_rate)
    nsc.model.train()
    random.shuffle(train_exs)

    # train_exs = train_exs[:1000]

    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(train_exs)

        for sentence in train_exs:
            y_onehot = torch.tensor([0., 1.]) if sentence.label == 1 else torch.tensor([1., 0.])

            nsc.model.zero_grad()
            probs = nsc.model(sentence)

            loss = loss_crit(probs, y_onehot)
            total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))

    return nsc
