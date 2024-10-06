# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).

# 1. Add positional encodings to the input (leave out until Part 1)
# 2. Use one or more TransformerLayers
# 3. Use Linear and LogSoftmax layers  to make log probability predictions
# Network should return log probs at the output layer (20x3 matrix) and attentions which are plotted in plots/
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, d_ff, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.d_internal = d_internal
        self.d_ff = d_ff
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.1)
        nn.init.xavier_uniform_(self.linear.weight)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal, d_ff) for _ in range(num_layers)])

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        list_attentions = []
        indices = self.embedding(indices)  # [seq len x d_model]

        for layer in self.layers:
            indices = layer(indices)
            list_attentions.append(layer.attention)

        indices = self.dropout(indices)
        log_probs = self.log_softmax(self.linear(indices))
        att_maps = torch.stack(list_attentions)

        return log_probs, att_maps


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.

# 1. self-attention (single headed is fine. you can use either backward-only or bidirectional attention)
# 2. residual connection
# 3. linear layer, non-linearity, linear layer  (feedforward layer)
# 4. final residual connection

# form queries, keys, and values matrices with linear layers, then use the queries and keys
# to compute attention over the sentence, then combine with the values.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal, d_ff):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super(TransformerLayer, self).__init__()
        self.d_internal = d_internal

        self.WQ = nn.Linear(d_model, d_internal)
        self.WK = nn.Linear(d_model, d_internal)
        self.WV = nn.Linear(d_model, d_internal)
        self.W0 = nn.Linear(d_internal, d_model)

        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)
        nn.init.xavier_uniform_(self.W0.weight)

        self.attention = torch.zeros((20, 20))

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        nn.init.xavier_uniform_(self.ff[0].weight)
        nn.init.xavier_uniform_(self.ff[2].weight)

    def forward(self, input_vecs: torch.Tensor):
        """
        :param input_vecs: a [seq len x d_model] tensor. This is our embeddings matrix X
        :return: a [seq len x d_model] tensor
        """

        Q = self.WQ(input_vecs)
        K = self.WK(input_vecs)
        V = self.WV(input_vecs)

        self_attention_output = self.self_attention(Q, K, V)

        prep_first_residual = self.W0(self_attention_output)
        first_residual = prep_first_residual + input_vecs

        ffn_output = self.ff(first_residual)

        second_residual = ffn_output + first_residual
        return second_residual

    def self_attention(self, queries, keys, values):
        scores = torch.matmul(queries, keys.transpose(-2, -1))  # 20x20 matrix

        stabilize_grads = scores / (self.d_internal ** 0.5)
        softmaxed = torch.nn.functional.softmax(stabilize_grads, dim=-1)
        self.attention = softmaxed
        return torch.matmul(softmaxed, values)


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train: list[LetterCountingExample], dev):

    # hyperparameters
    num_epochs = 20

    # model hyperparameters
    d_model = 20  # dimension of embeddings
    d_internal = 128  # dimension of Q and K for simpler computation
    d_ff = 2048  # dimension of feedforward layer
    num_layers = 1

    # model parameters
    vocab_size = 27
    num_positions = 20
    num_classes = 3
    loss_fcn = nn.NLLLoss()

    model = Transformer(vocab_size, num_positions, d_model, d_internal, d_ff, num_classes, num_layers)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # only for debugging
    # train = train[:1000]

    for epoch in range(0, num_epochs):
        loss_this_epoch = 0.0

        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)

        for ex_idx in ex_idxs:
            optimizer.zero_grad()
            log_probs, attn_maps = model(train[ex_idx].input_tensor)

            loss = loss_fcn(log_probs, train[ex_idx].output_tensor)
            loss.backward()
            optimizer.step()

            loss_this_epoch += loss
        print("Loss on epoch %i: %f" % (epoch, loss_this_epoch))
    model.eval()
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
