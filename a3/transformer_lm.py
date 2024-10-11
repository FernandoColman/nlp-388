# models.py
import math

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from a3.utils import Indexer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int = 20, batched=False):
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


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class TransformerLM(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, batched, max_seq_len, vocab_size=27, dropout=0.1):
        super(TransformerLM, self).__init__()

        # Embeddings and Positional Encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embeds = PositionalEncoding(d_model, max_seq_len, batched)

        # Encoder Layers
        encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        # Output Layer
        self.output = nn.Linear(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # Adding Dropout
        self.dropout = nn.Dropout(dropout)

        # Dimensions
        self.d_model = d_model

        self.init_weights()

    def init_weights(self):
        # initialize embeddings and output
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.positional_embeds.emb.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)

        # initialize encoder (self-attention and feedforward)
        for layer in self.encoder.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.constant_(layer.self_attn.in_proj_bias, 0)
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.constant_(layer.linear1.bias, 0)
            nn.init.xavier_uniform_(layer.linear2.weight)
            nn.init.constant_(layer.linear2.bias, 0)

    def forward(self, src):
        # Embedding + positional encoding
        embeds = self.embedding(src) * math.sqrt(self.d_model)  # why are we multiplying?
        pos_embeds = self.positional_embeds(embeds)

        # Masking so the model does not look at future characters
        mask = torch.triu(torch.ones(len(pos_embeds), len(pos_embeds)) * float('-inf'), diagonal=1)

        # Pass through transformer encoder. Will pass through all layers
        encoded = self.encoder(pos_embeds, mask)

        # Dropout
        dropped = self.dropout(encoded)

        # Output layer (linear transformation to vocab size)
        log_probs = self.log_softmax(self.output(dropped))
        return log_probs


class NeuralLanguageModel(LanguageModel):
    def __init__(self, d_model, num_heads, num_layers, d_ff, batched, max_seq_len, vocab_index: Indexer):
        super(NeuralLanguageModel, self).__init__()
        self.max_seq_len = max_seq_len
        self.vocab_index = vocab_index
        self.model = TransformerLM(d_model, num_heads, num_layers, d_ff, batched, max_seq_len)

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
                Returns a log probability distribution over the next characters given a context.
                The log should be base e

                NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
                layers in TransformerEncoder).
                :param context: the string context that the LM conditions on
                :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
                """

        self.model.eval()
        ctx_idx = [self.vocab_index.index_of(c) for c in context]
        probs = self.model.forward(torch.tensor(ctx_idx))[-1]

        output = probs.squeeze()

        ndarray = output.detach().numpy()
        return ndarray

    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
                Scores a bunch of characters following context. That is, returns
                log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
                The log should be base e

                NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
                layers in TransformerEncoder).
                :param next_chars:
                :param context:
                :return: The float probability
                """

        self.model.eval()

        if context == "":
            context = " "

        log_prob = 0.0
        if len(next_chars) > self.max_seq_len:
            # add padding to calculate last characters
            next_chars += " " * (self.max_seq_len - len(next_chars) % self.max_seq_len)

            # starts at 20 and increases by 20
            chunk_count = self.max_seq_len

            while chunk_count < len(next_chars):
                # grabs a chunk of the training data with length max_seq_len - 1 (to account for the starting space))
                chunk = " " + next_chars[chunk_count - self.max_seq_len: chunk_count - 1]

                for i in range(len(chunk)):
                    ctx_idx = [self.vocab_index.index_of(c) for c in context + chunk[:i]]
                    probs = self.model.forward(torch.tensor(ctx_idx))[-1].squeeze()
                    log_prob += probs[self.vocab_index.index_of(chunk[i])]
                chunk_count += self.max_seq_len

        else:
            for i in range(len(next_chars)):
                ctx_idx = [self.vocab_index.index_of(c) for c in context + next_chars[:i]]
                probs = self.model.forward(torch.tensor(ctx_idx))[-1].squeeze()
                log_prob += probs[self.vocab_index.index_of(next_chars[i])]

        return log_prob.detach().numpy()


def train_lm(args, train_text: str, dev_text, vocab_index: Indexer) -> NeuralLanguageModel:
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    # for debugging
    # train_text = train_text[:100]

    # padding
    train_text += " " * (20 - len(train_text) % 20)

    # training hyperparameters
    num_epochs = 15

    # model hyperparameters
    learning_rate = 0.0001
    d_model = 256           # dimension of embeddings
    num_heads = 2               # number of heads in the multi-head attention
    d_ff = 64             # dimension of feedforward layer
    num_layers = 2
    max_seq_len = 20
    batched = False

    loss_fcn = nn.NLLLoss()
    nlm = NeuralLanguageModel(d_model, num_heads, num_layers, d_ff, batched, max_seq_len, vocab_index)
    nlm.model.zero_grad()
    nlm.model.train()
    optimizer = optim.Adam(nlm.model.parameters(), lr=learning_rate)

    # padding
    train_text += " " * (max_seq_len - len(train_text) % max_seq_len)

    for epoch in range(0, num_epochs):
        loss_this_epoch = 0.0

        train_seq = max_seq_len  # starts at 20 and increases by 20
        while train_seq < len(train_text):
            optimizer.zero_grad()

            # grabs a chunk of the training data with length max_seq_len
            train_ex = " " + train_text[train_seq - max_seq_len: train_seq - 1]

            # converts the chunk to a list of indices
            ex_idx = [vocab_index.index_of(c) for c in train_ex]
            # print("Input Words: \""+ train_ex+ "\". Indices: ", ex_idx)

            # trains the model on the list of indices
            log_probs = nlm.model(torch.tensor(ex_idx))
            # print("Log Probs: ", log_probs)

            # sets goal to be the next character in the training data
            gold_ex = train_text[train_seq - max_seq_len:train_seq]
            gold_ex_idx = [vocab_index.index_of(c) for c in gold_ex]
            # print("Gold Words:  \"" + gold_ex + "\". Indices: ", gold_ex_idx)

            loss = loss_fcn(log_probs, torch.tensor(gold_ex_idx))
            loss.backward()
            optimizer.step()

            loss_this_epoch += loss.item()
            train_seq += max_seq_len
        print("Loss on epoch %i: %f" % (epoch + 1, loss_this_epoch))
    nlm.model.eval()
    return nlm
