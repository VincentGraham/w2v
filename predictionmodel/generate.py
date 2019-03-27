from load_vocab import load_word2vec_model, load_word2vec_vocab
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils import checkpoint
import numpy as np
from rearrange import check_case, tokenize
import gc
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


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, loss, opt=None):
        self.loss = loss
        self.generator = generator
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = loss / norm

        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm


def run_epoch(data_iter, model, optim, print_every=50):
    """Standard Training and Logging Function"""
    # TODO add on the fly tensorizing of data instead of initial
    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0
    for i, batch in enumerate(data_iter, 1):
        loss, o = model.forward(batch.src, batch.trg, batch.src_mask,
                                batch.trg_mask, batch.src_lengths,
                                batch.trg_lengths)
        out, _, pre_output = o
        loss = SimpleLossCompute(model.module.generator, criterion, optim)
        total_loss += float(loss)
        total_tokens += batch.ntokens
        print_tokens += batch.ntokens

        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.nseqs, print_tokens / elapsed))
            start = time.time()
            print_tokens = 0

    return math.exp(total_loss / float(total_tokens))


def data_gen(sentence,
             num_words=11,
             batch_size=1,
             num_batches=1,
             length=10,
             pad_index=1,
             sos_index=1):
    """Generate random data for a src-tgt copy task."""
    for i in range(num_batches):
        data = torch.from_numpy(
            reverse_lookup_words(sentence, length, pad_index, SRC.vocab))
        data[:, 0] = sos_index
        data = data.cuda() if USE_CUDA else data
        src = data[:, 1:]
        trg = data
        src_lengths = [length - 1]
        trg_lengths = [length]
        yield Batch((src, src_lengths), (trg, trg_lengths),
                    pad_index=pad_index)


def greedy_decode(model,
                  src,
                  src_mask,
                  src_lengths,
                  max_len=100,
                  sos_index=1,
                  eos_index=None):
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
            prob = model.module.generator(pre_output[:, -1])

        _, next_word = torch.max(prob, dim=1)
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


def reverse_lookup_words(x, length, token, vocab=None):
    z = [0]
    if vocab is not None:
        z += [vocab.itos.index(i) for i in x.split(' ')]
    while len(z) < length:
        z += ([token])
    return np.asarray([z])


def lookup_words(x, vocab=None):
    if vocab is not None:
        x = [vocab.itos[i] for i in x]

    return [str(t) for t in x]


def lookup_words_full_vocab(x):
    x = [vocab[i] for i in x]
    return [str(x) for _ in x]


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

    model.eval()
    count = 0
    print()

    if src_vocab is not None and trg_vocab is not None:
        src_eos_index = src_vocab.stoi[EOS_TOKEN]
        trg_sos_index = trg_vocab.stoi[SOS_TOKEN]
        trg_eos_index = trg_vocab.stoi[EOS_TOKEN]
    else:
        src_eos_index = None
        trg_sos_index = 1
        trg_eos_index = None

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
        print("Src : ", " ".join(lookup_words(src, vocab=src_vocab)))
        print("Trg : ", " ".join(lookup_words(trg, vocab=trg_vocab)))
        print("Pred: ", " ".join(lookup_words(result, vocab=trg_vocab)))
        print()

        count += 1
        if count == n:
            break


def plot_perplexity(perplexities):
    """plot perplexities"""
    plt.title("Perplexity per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.plot(perplexities)


from torchtext import data, datasets, vocab

if True:
    import json

    def token(text):
        return [
            x for x in tokenize(text)[0].replace('(', "").replace(")",
                                                                  "").split()
        ]

    UNK_TOKEN = "<u>"
    PAD_TOKEN = "<p>"
    SOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    LOWER = True

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

    MAX_LEN = 25  # NOTE: we filter out a lot of sentences for speed

    data_fields = [('sentence', SRC), ('article', TRG)]

    train_data, valid_data = data.TabularDataset.splits(
        path="/mounted/data",
        train='test.csv',
        validation='test.csv',
        format="csv",
        fields=data_fields,
        filter_pred=lambda x: len(vars(x)['sentence']) <= MAX_LEN and len(
            vars(x)['article']) <= 50)
    MIN_FREQ = 2  # NOTE: we limit the vocabulary to frequent words for speed
    VOCAB = vocab.Vectors('model.txt', cache='/mounted/data')
    SRC.build_vocab(train_data, vectors=VOCAB, min_freq=MIN_FREQ)
    TRG.build_vocab(train_data, vectors=VOCAB, min_freq=MIN_FREQ * 2)

    PAD_INDEX = TRG.vocab.stoi[PAD_TOKEN]

# TODO: add custom tokens using the rearrange .py file


def print_data_info(train_data, valid_data, src_field, trg_field):
    """ This prints some useful stuff about our data sets. """

    print("Data set sizes (number of sentence pairs):")
    print('train', len(train_data))
    print('valid', len(valid_data))

    print("First training example:")
    print("src:", " ".join(vars(train_data[0])['sentence']))
    print("trg:", " ".join(vars(train_data[0])['article']), "\n")

    print("Most common words (src):")
    print(
        "\n".join(
            ["%10s %10d" % x for x in src_field.vocab.freqs.most_common(15)]),
        "\n")
    print("Most common words (trg):")
    print(
        "\n".join(
            ["%10s %10d" % x for x in trg_field.vocab.freqs.most_common(15)]),
        "\n")

    print("First 10 words (src):")
    print(
        "\n".join('%02d %s' % (i, t)
                  for i, t in enumerate(src_field.vocab.itos[:10])), "\n")
    print("First 10 words (trg):")
    print(
        "\n".join('%02d %s' % (i, t)
                  for i, t in enumerate(trg_field.vocab.itos[:10])), "\n")

    print("Number of German words (types):", len(src_field.vocab))
    print("Number of English words (types):", len(trg_field.vocab), "\n")


print_data_info(train_data, valid_data, SRC, TRG)

train_iter = data.BucketIterator(
    train_data,
    batch_size=1,
    train=True,
    sort_within_batch=True,
    sort_key=lambda x: len(x.sentence),
    repeat=False)


def wrap_data(data):
    return torch.Tensor(data)


def rebatch(pad_idx, batch):
    """Wrap torchtext batch into our own Batch class for pre-processing"""
    return Batch(batch.sentence, batch.article, pad_idx)


def train(model, num_epochs=10, lr=0.0003, print_every=100):

    # optionally add label smoothing; see the Annotated Transformer
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    model.cuda()
    dev_perplexities = []

    for epoch in range(num_epochs):

        print("Epoch", epoch)
        model.train()
        train_perplexity = run_epoch(
            (rebatch(PAD_INDEX, b) for b in train_iter),
            model,
            optim,
            print_every=print_every)

    return dev_perplexities


vocab = load_word2vec_vocab()
pret = load_word2vec_model()

model = make_model(
    len(SRC.vocab),
    len(TRG.vocab),
    emb_size=500,
    hidden_size=256,
    num_layers=2,
    dropout=0.1)

model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()

dev_perplexities = train(model, print_every=100, num_epochs=100)

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
