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
from adasoft import *
from models import *

USE_CUDA = False
DEVICE = torch.device('cuda:3')  # or set to 'cpu'
DEVICE1 = torch.device('cuda:1')
DEVICE2 = torch.device('cuda:2')
DEVICE3 = torch.device('cuda:0')

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
MAX_LEN = 10


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, src, trg, pad_index=0):
        src, src_lengths = src
        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)

        self.trg = None
        self.trg_y = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        if trg is not None:
            trg, trg_lengths = trg
            self.trg = trg[:, :-1]
            self.trg_lengths = trg_lengths
            self.trg_raw = self.trg.tolist()
            self.trg_lengths_raw = self.trg_lengths.tolist()
            self.trg_y = trg[:, 1:]
            self.trg_mask = (self.trg_y != pad_index)
            self.ntokens = (self.trg_y != pad_index).data.sum().item()

        if USE_CUDA:
            self.src = self.src.cuda()
            self.src_mask = self.src_mask.cuda()

            if trg is not None:
                self.trg = self.trg.cuda()
                self.trg_y = self.trg_y.cuda()
                self.trg_mask = self.trg_mask.cuda()


def run_epoch(data_iter, model, print_every=50, optim=None):
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0
    print(len(list(data_iter)))
    for i, batch in enumerate(data_iter, 1):
        # for obj in gc.get_objects():
        #     if (not os.path.isdir(str(obj) and not isinstance(obj, io.IOBase))
        #             and torch.is_tensor(obj)) or (os.path.isdir(
        #                 str(obj.data) and not isinstance(obj, io.IOBase)
        #                 and hasattr(obj, 'data')
        #                 and torch.is_tensor(obj.data))):
        #         # print(type(obj), obj.size())
        #         del obj
        #         gc.collect()
        batch = rebatch(PAD_INDEX, batch)
        out, _, pre_output, = model.forward(
            batch.src, batch.trg, batch.src_mask, batch.trg_mask,
            batch.src_lengths, batch.trg_lengths, batch.trg_y)

        # loss = loss_compute(pre_output, batch.trg_y, batch.nseqs)
        # m = AdaptiveSoftmax(256, [2000, 10000])
        # m = FacebookAdaptiveSoftmax(
        #     len(vocab), 256, [2000, 10000], dropout=0.1)
        # criterion = FacebookAdaptiveLoss(PAD_INDEX)
        criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
        criterion = nn.MSELoss()

        # x = pre_output.view(-1, pre_output.size()[1])
        # y = batch.trg_y.contiguous().view(
        #     batch.trg_y.size()[0] * batch.trg_y.size()[1])

        # print(x.size(), y.size(), batch.trg_y.size())

        def fix_target(tens):
            out = []
            for list in tens:
                words = []
                for word_idx in list:
                    word = vocab[word_idx]
                    words.append(word)  # replace with below for non-nllloss
                    # words.append(pret[word])  # the embedding
                out.append(words)
            return torch.Tensor(out)

        new_trg = fix_target(batch.trg_raw)

        loss = criterion(pre_output, new_trg)
        total_tokens += batch.ntokens
        print_tokens += batch.ntokens
        loss.backward()
        optim.step()
        optim.zero_grad()
        loss.detach()
        total_loss += loss.item()

        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.nseqs, print_tokens / elapsed))
            start = time.time()
            print_tokens = 0

    return math.exp(total_loss / float(total_tokens))


def reverse_lookup_words(x, length, token, vocab=None):

    z = [0]
    if vocab is not None:
        z += [vocab.index(i) for i in x.split(' ')]
    while len(z) < length:
        z += ([token])
    return np.asarray([z])


def lookup_words_full_vocab(x):
    x = [vocab[i] for i in x]
    return x


def lookup_words_full_vocab_from_embeddings(x):
    return [pret.similar_by_vector(i)[0][0] for i in x]


def reverse_lookup_words_full_vocab(x, length, token):

    z = [0]
    if vocab is not None:
        z += [vocab.index(i) for i in x.split(' ')]  # i is a sentence
    while len(z) < length:
        z += ([token])
    z += [EOS_INDEX]
    return np.asarray([z])


# def reverse_lookup_words_full_vocab_with_pre_embedding(x, length, token):
#     # passed a Tensor.tolist() -> [[]]
#     z =  # [[]]


def data_gen(sentence, article, num_words=11, num_batches=1, length=MAX_LEN):
    for i in range(num_batches):
        data = torch.from_numpy(
            reverse_lookup_words_full_vocab(
                sentence,
                length,
                PAD_INDEX,
            ))
        data[:, 0] = SOS_INDEX
        data = data.cuda() if USE_CUDA else data
        src = data[:, 1:]
        target = torch.from_numpy(
            reverse_lookup_words_full_vocab(article, length, PAD_INDEX))
        target = target.cuda() if USE_CUDA else target
        trg = target[:]
        src_lengths = [length - 1]
        trg_lengths = [length]
        yield Batch((src, src_lengths), (trg, trg_lengths),
                    pad_index=PAD_INDEX)


def data_gen_single(sentence,
                    article,
                    num_words=11,
                    num_batches=1,
                    length=MAX_LEN):
    data = torch.from_numpy(
        reverse_lookup_words_full_vocab(
            sentence,
            length,
            PAD_INDEX,
        ))
    data[:, 0] = SOS_INDEX
    data = data.cuda() if USE_CUDA else data
    src = data[:, 1:]
    target = torch.from_numpy(
        reverse_lookup_words_full_vocab(article, length, PAD_INDEX))
    target = target.cuda() if USE_CUDA else target
    trg = target[:]
    src_lengths = [length - 1]
    trg_lengths = [length]
    return Batch(
        (src, src_lengths), (trg, trg_lengths),
        pad_index=PAD_INDEX)  # src is torch.Tensor and src_lengths is []


def greedy_decode(model,
                  src,
                  src_mask,
                  src_lengths,
                  max_len=100,
                  sos_index=SOS_INDEX,
                  eos_index=EOS_INDEX):
    """Greedily decode a sentence."""

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask,
                                                     src_lengths)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)

    output = []
    attention_scores = []
    hidden = None
    for i in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output = model.decode(encoder_hidden,
                                                   encoder_final, src_mask,
                                                   prev_y, trg_mask, hidden)

            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
        next_word = pret.similar_by_vector(pre_output[0][0].tolist())
        next_word = next_word.data.item()
        output.append(next_word)
        prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
        attention_scores.append(model.decoder.attention.alphas.cpu().numpy())

    output = np.array(output)

    # cut off everything starting from </s>
    # (only when eos_index provided)
    if eos_index is not None:
        first_eos = np.where(output == eos_index)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]

    return output, np.concatenate(attention_scores, axis=1)


def lookup_words(x, vocab=None):
    if vocab is not None:
        x = [vocab[i] for i in x]

    return [str(t) for t in x]


def print_examples(example_iter,
                   model,
                   n=2,
                   max_len=100,
                   sos_index=1,
                   src_eos_index=None,
                   trg_eos_index=None,
                   src_vocab=None,
                   trg_vocab=None):
    """Prints N examples. Assumes batch size of 1."""

    def fix_target(tens):
        out = []
        for list in tens:
            words = []
            for word_idx in list:
                word = vocab[word_idx]
                words.append(word)  # replace with below for non-nllloss
                # words.append(pret[word])  # the embedding
            out.append(words)
        return torch.Tensor(out)

    model.eval()
    count = 0
    print()
    with torch.no_grad():
        if src_vocab is not None and trg_vocab is not None:
            src_eos_index = EOS_INDEX
            trg_sos_index = SOS_INDEX
            trg_eos_index = EOS_INDEX
        else:
            src_eos_index = None
            trg_sos_index = 1
            trg_eos_index = None

        if isinstance(example_iter, Batch):
            batch = example_iter
            src = batch.src.cpu().numpy()[0, :]
            trg = fix_target(batch.trg_raw).cpu().numpy()[0, :]

            # remove </s> (if it is there)
            result, _ = greedy_decode(
                model,
                batch.src,
                batch.src_mask,
                batch.src_lengths,
                max_len=max_len,
                sos_index=trg_sos_index,
                eos_index=trg_eos_index)
            print("Example #%d" % (i + 1))
            print("Src : ", " ".join(lookup_words_full_vocab(src)))
            print("Trg : ", " ".join(
                lookup_words_full_vocab_from_embeddings(trg)))
            print("Pred: ", " ".join(
                lookup_words_full_vocab_from_embeddings(result)))
            print()

            return " ".join(
                lookup_words_full_vocab_from_embeddings(
                    result, vocab=trg_vocab))
        else:
            for i, batch in enumerate(example_iter):
                src = batch.src.cpu().numpy()[0, :]
                trg = batch.trg_y.cpu().numpy()[0, :]

                # remove </s> (if it is there)

                result, _ = greedy_decode(
                    model,
                    batch.src,
                    batch.src_mask,
                    batch.src_lengths,
                    max_len=max_len,
                    sos_index=trg_sos_index,
                    eos_index=trg_eos_index)
                print("Example #%d" % (i + 1))
                print("Src : ", " ".join(lookup_words_full_vocab(src)))
                print("Trg : ", " ".join(
                    lookup_words_full_vocab_from_embeddings(trg)))
                print(
                    "Pred: ", " ".join(
                        lookup_words_full_vocab_from_embeddings(result)))
                print()

                count += 1
                if count == n:
                    break
                return " ".join(
                    lookup_words_full_vocab_from_embeddings(
                        result, vocab=trg_vocab))


def plot_perplexity(perplexities):
    """plot perplexities"""
    plt.title("Perplexity per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.plot(perplexities)


from torchtext import data, datasets
from torchtext import vocab as vv

if True:
    import json

    def token(text):
        return [
            x for x in tokenize(text)[0].replace('(', "").replace(")",
                                                                  "").split()
        ]

    # we include lengths to provide to the RNNs
    SRC = data.Field(
        tokenize=token,
        batch_first=True,
        lower=LOWER,
        include_lengths=True,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        init_token=SOS_TOKEN,
        eos_token=EOS_TOKEN)
    TRG = data.Field(
        batch_first=True,
        lower=LOWER,
        include_lengths=True,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        init_token=SOS_TOKEN,
        eos_token=EOS_TOKEN)

    MAX_LEN = 30  # NOTE: we filter out a lot of sentences for speed

    data_fields = [('sentence', SRC), ('article', TRG)]

    train_data, valid_data = data.TabularDataset.splits(
        path="/mounted/data",
        train='test.csv',
        validation='test.csv',
        format="csv",
        fields=data_fields)

    MIN_FREQ = 1  # NOTE: we limit the vocabulary to frequent words for speed
    VOCAB = vv.Vectors('model.txt', cache='/mounted/data')
    SRC.build_vocab(train_data, vectors=VOCAB, min_freq=MIN_FREQ)
    TRG.build_vocab(train_data, vectors=VOCAB, min_freq=MIN_FREQ)
    print(PAD_INDEX, SOS_INDEX, EOS_INDEX)

# print_data_info(train_data, valid_data, SRC, TRG)

train_iter = data.BucketIterator(
    train_data,
    batch_size=64,
    train=True,
    sort_within_batch=True,
    sort_key=lambda x: len(x.sentence),
    repeat=False,
    device=DEVICE)


def rebatch(pad_idx, batch):
    """Wrap torchtext batch into our own Batch class for pre-processing"""
    # for t in batch.sentence:
    #     t.to(DEVICE)
    # for t in batch.article:
    #     t.to(DEVICE)
    return Batch(batch.sentence, batch.article, pad_idx)


def train(model, num_epochs=10, lr=0.0003, print_every=100):

    # optionally add label smoothing; see the Annotated Transformer
    criterion = nn.NLLLoss(reduction="sum", ignore_index=PAD_INDEX)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    dev_perplexities = []

    for epoch in range(num_epochs):

        print("Epoch", epoch)
        model.train()
        train_perplexity = run_epoch(
            train_iter, model, print_every=1, optim=optim)

        model.eval()
        with torch.no_grad():
            print_examples((rebatch(PAD_INDEX, x) for x in valid_iter),
                           model,
                           n=3,
                           src_vocab=vocab,
                           trg_vocab=vocab)

            dev_perplexity = run_epoch(
                (rebatch(PAD_INDEX, b) for b in valid_iter),
                model,
            )
            print("Validation perplexity: %f" % dev_perplexity)
            dev_perplexities.append(dev_perplexity)

    return dev_perplexities


model = make_model(
    len(vocab),
    len(vocab),
    emb_size=500,
    hidden_size=256,
    num_layers=3,
    dropout=0.1)

model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()

dev_perplexities = train(model, num_epochs=100)

torch.save(model.state_dict(), '/mounted/data/torch/parallel_model')
torch.save(model.module.state_dict(), '/mounted/data/torch/model')


def load_model():
    model = make_model(
        len(SRC.vocab),
        len(TRG.vocab),
        emb_size=500,
        hidden_size=1500,
        num_layers=3,
        dropout=0.2)
    model.load_state_dict(torch.load('/mounted/data/torch/model'))
    model.eval()
    return model


def load_dataparallel_model():
    from collections import OrderedDict
    model = make_model(
        len(SRC.vocab),
        len(TRG.vocab),
        emb_size=500,
        hidden_size=1500,
        num_layers=3,
        dropout=0.2)
    model.load_state_dict(torch.load('/mounted/data/torch/parallel_model'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model
