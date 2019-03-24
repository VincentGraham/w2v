# -*- coding: utf-8 -*-

import gensim
import logging
import multiprocessing
import os
import re
import sys
import pickle
from pattern.text.en import tokenize
from time import time
from gensim.models.phrases import Phrases, Phraser
import psutil

assert gensim.models.word2vec.FAST_VERSION > -1

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load():
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
        '/Users/vincent/Desktop/mounted/data/word2vec2.bin', binary=True)
    print("Converting Model to weight vectors...")
    # return torch.FloatTensor(word2vec_model.vectors)
    return word2vec_model


def load_vocab():
    with open("/Users/vincent/Desktop/mounted/data/vocab2.pkl", "rb") as file:
        vocab_list = pickle.load(file)
        file.close()
    return vocab_list


if __name__ == '__main__':
    model = load()
    print(psutil.cpu_percent())
    vocab = load_vocab()
    print(vocab)

    while (True):
        pos = []
        inp = ""
        while len(pos) < 2:
            inp = input("Enter a word: ")
            if inp not in vocab:
                continue
            if 'stop' in inp or inp == "":
                break
            pos.append(inp)
        if inp not in vocab:
            continue
        # print(model.most_similar(positive=pos))
        print(model.similarity(pos[0], pos[1]))
