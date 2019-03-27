from load_vocab import load_word2vec_model, load_word2vec_vocab, load_csv_for_tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from rearrange import check_case, tokenize
import logging
import gc
import io
import os
import psutil
import random
from adasoft import AdaptiveLoss, AdaptiveSoftmax, FacebookAdaptiveSoftmax, FacebookAdaptiveLoss

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

USE_CUDA = False
DEVICE = torch.device('cpu')

# or set to 'cpu'
FAST = False
ACCUMULATION = 32
LOWER = False  # CASE SENSITIVE

vocab = load_word2vec_vocab()
pret = load_word2vec_model()

logging.info("Loaded gensim models")
UNK_TOKEN = "<u>"
PAD_TOKEN = "<p>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

PAD_INDEX = vocab.index(PAD_TOKEN)
SOS_INDEX = vocab.index(SOS_TOKEN)
EOS_INDEX = vocab.index(EOS_TOKEN)
UNK_INDEX = vocab.index(UNK_TOKEN)
MAX_LEN = 10


class EncoderDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_embed,
                 trg_embed,
                 generator,
                 nh=0,
                 vs=0):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        # self.generator = nn.AdaptiveLogSoftmaxWithLoss(
        #     nh, vs, cutoffs=[round(vs / 30), 3 * round(vs / 30)], div_value=4)

    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths,
                trg_y):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)

        out, _, pre_output = self.decode(encoder_hidden, encoder_final,
                                         src_mask, trg, trg_mask)

        # output, loss = self.generator(pre_output, trg_y)
        # return out, _, pre_output, output, loss
        return out, _, pre_output

    def encode(self, src, src_mask, src_lengths):
        return self.encoder(self.src_embed(src), src_mask, src_lengths)

    def decode(self,
               encoder_hidden,
               encoder_final,
               src_mask,
               trg,
               trg_mask,
               decoder_hidden=None):
        return self.decoder(
            self.trg_embed(trg),
            encoder_hidden,
            encoder_final,
            src_mask,
            trg_mask,
            hidden=decoder_hidden)


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class GeneratorTest(nn.Module):
    def __init__(self, nh, vs):
        super(GeneratorTest, self).__init__()
        self.out = nn.AdaptiveLogSoftmaxWithLoss(
            nh, vs, cutoffs=[round(vs / 30), 3 * round(vs / 30)], div_value=4)

    def forward(x, y):
        x = x.view(-1, x.size()[2])
        y = y.view(y.size()[0] * y.size()[1])
        return self.out(x, y)


class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout)

    def forward(self, x, mask, lengths):
        """
        Applies a bidirectional GRU to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        output, final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # we need to manually concatenate the final states for both directions
        fwd_final = final[0:final.size(0):2]
        bwd_final = final[1:final.size(0):2]
        final = torch.cat([fwd_final, bwd_final],
                          dim=2)  # [num_layers, batch, 2*dim]

        return output, final


def make_model(src_vocab,
               tgt_vocab,
               emb_size=500,
               hidden_size=1024,
               num_layers=3,
               dropout=0.1,
               pret=None):
    "Helper: Construct a model from hyperparameters."

    attention = BahdanauAttention(hidden_size)

    embedding1 = nn.Embedding.from_pretrained(torch.FloatTensor(pret.vectors))
    embedding2 = nn.Embedding.from_pretrained(torch.FloatTensor(pret.vectors))

    #TODO change the embedding to a linear layer that takes model[word] vectors.
    # pa
    embedding1.weight.requires_grad = False
    embedding2.weight.requires_grad = False

    model = EncoderDecoder(
        Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(
            emb_size,
            hidden_size,
            attention,
            num_layers=num_layers,
            dropout=dropout),
        embedding1,
        embedding2,
    )

    return model
