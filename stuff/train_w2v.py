# -*- coding: utf-8 -*-
# cython: language_level=3
import nltk
nltk.data.path = ['/Users/vincent/Desktop/mounted/nltk_data']

import cython
import gensim
import logging
import multiprocessing
import os
import re
import sys
import pickle
import random
import timeit

from pattern.text.en import tokenize
from time import time

# TODO check O notation and improve speed of iterating

assert gensim.models.word2vec.FAST_VERSION > -1

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        start_time = timeit.default_timer()
        for root, dirs, files in os.walk(self.dirname):
            for filename in files:
                file_path = root + '/' + filename
                if filename == '.DS_Store':
                    continue
                for line in open(file_path, encoding='utf-8', errors='ignore'):
                    sline = line.strip()
                    if sline == "":
                        continue
                    rline = cleanhtml(sline)
                    toks = tokenize(rline)
                    if len(toks) == 1:  # almost always unless empty list
                        while (len(toks[0].split()) < 5):
                            toks[0] += ' <p>'
                        if random.randint(0, 100) < 1:
                            to_replace = toks[0].split()
                            to_replace[random.randint(
                                0,
                                len(to_replace) - 1)] = '<u>'
                            toks = to_replace
                    tokenized_line = ' '.join(toks)
                    is_alpha_word_line = ['<s>'] + [
                        word for word in tokenized_line.split() if (
                            (word.isalpha() or word.isdigit()) or word == '<p>'
                            or word == '<u>' or '-' in word)
                    ] + ['</s>']
                    is_alpha_word_line = is_alpha_word_line
                    yield is_alpha_word_line
            print('Done at {}'.format(root))
            print(timeit.default_timer() - start_time)
        print(timeit.default_timer() - start_time)


if __name__ == '__main__':
    data_path = '/Users/vincent/Desktop/mounted/enwiki'
    begin = time()
    sentences = None
    sentences = MySentences(data_path)

    if len(sys.argv) >= 2:
        epochs = int(sys.argv[1])
        a = float(sys.argv[2])
    else:
        epochs = 25
        a = 0.001

    model = gensim.models.Word2Vec(
        sentences,
        size=500,
        window=15,
        min_count=10,
        workers=multiprocessing.cpu_count(),
        alpha=a,
        min_alpha=1e-5,
        iter=epochs)
    model.save("/Users/vincent/Desktop/mounted/data/word2vec.model")
    model.wv.save_word2vec_format(
        "/Users/vincent/Desktop/mounted/data/word2vec.bin", binary=True)
    with open("/Users/vincent/Desktop/mounted/data/vocab.pkl", 'wb') as file:
        pickle.dump(list(model.wv.vocab.keys()), file)
    file.close()

    end = time()
    print("Total procesing time: %d seconds" % (end - begin))
    print("Testing custom tokens and sample word")
    print('<p>', model.wv.most_similar('<p>'), '\n')
    print('<u>', model.wv.most_similar('<u>'), '\n')
    print('<s>', model.wv.most_similar('<s>'), '\n')
    print('</s>', model.wv.most_similar('</s>'), '\n')
    print('alaska', model.wv.most_similar('alaska'))
