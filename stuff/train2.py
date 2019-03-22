# -*- coding: utf-8 -*-
# cython: language_level=3
# import nltk
# nltk.data.path = ['/Users/vincent/Desktop/mounted/nltk_data']

import cython
import gensim
import logging
import multiprocessing as mp
import os
import re
import sys
import pickle
import random
import timeit
import threading
import queue

from pympler import summary, muppy
from multiprocessing import Process, Queue, Pool
from rearrange import get_sentences, main
from time import time

# TODO check O notation and improve speed of iterating

assert gensim.models.word2vec.FAST_VERSION > -1
assert gensim.models.doc2vec.FAST_VERSION > -1

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

compiled_expression = re.compile('<.*?>')


def cleanhtml(raw_html):
    global compiled_expression
    cleantext = re.sub(compiled_expression, ' ', raw_html)
    return cleantext


q = Queue()
output = Queue()
THREADS = 24


class MyThread(threading.Thread):
    def __init__(self, queue2, output, *args, **kwargs):
        self.queue = queue2
        self.output = output
        super().__init__(*args, **kwargs)

    def run(self):
        while True:
            try:
                pass
            except queue.Empty:
                return
            main(self.queue, self.output)
            # self.queue.task_done()


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        processes = []

        for root, dirs, files in os.walk(self.dirname):
            paths = []
            start_time = timeit.default_timer()
            [
                q.put(root + '/' +
                      filename if filename != '.DS_Store' else filename)
                for filename in files
            ]
        """MultiThreaded"""
        # for _ in range(24):
        #     pr = MyThread(q, output)
        #     pr.start()
        #     processes.append(pr)
        """MultiProcess"""
        for n in range(THREADS):
            p = Process(target=main, args=(q, output))
            processes.append(p)
            p.start()

        while not output.empty() or not q.empty():

            out = output.get()
            yield out
        #     print('Done at {}'.format(root))
        #     print(timeit.default_timer() - start_time)
        # print(timeit.default_timer() - start_time)

    # def __iter__(self):
    #     start_time = timeit.default_timer()
    #     for root, dirs, files in os.walk(self.dirname):
    #         paths = []
    #         for filename in files:
    #             file_path = root + '/' + filename
    #             if filename == '.DS_Store':
    #                 continue
    #             process = Process(target=get_sentences, args=[file_path])
    #             process.start()
    #             process.join()
    #             x = output.get()
    #             print(x)
    #             yield x

    #         print('Done at {}'.format(root))
    #         print(timeit.default_timer() - start_time)
    #     print(timeit.default_timer() - start_time)


if __name__ == '__main__':
    data_path = '/Users/vincent/Desktop/mounted/enwiki/AA'
    begin = time()
    sentences = None
    sentences = MySentences(data_path)
    if len(sys.argv) >= 2:
        epochs = int(sys.argv[1])
        a = float(sys.argv[2])
    else:
        epochs = 1
        a = 0.001

    model = gensim.models.Word2Vec(
        sentences,
        size=500,
        window=15,
        min_count=4,
        workers=mp.cpu_count() / 3,
        alpha=a,
        min_alpha=1e-7,
        iter=epochs,
        sg=1,
        batch_words=7500)
    model.save("/Users/vincent/Desktop/mounted/data/word2vec2.model")
    model.wv.save_word2vec_format(
        "/Users/vincent/Desktop/mounted/data/word2vec2.bin", binary=True)
    with open("/Users/vincent/Desktop/mounted/data/vocab2.pkl", 'wb') as file:
        pickle.dump(list(model.wv.vocab.keys()), file)
    file.close()

    end = time()
    print(len(model.wv.vocab.keys()))
    print("Total procesing time: %d seconds" % (end - begin))
    print("Testing custom tokens and sample word")
    print('<p>', model.wv.most_similar('<p>'), '\n')
    print('<u>', model.wv.most_similar('<u>'), '\n')
    print('<s>', model.wv.most_similar('<s>'), '\n')
    print('</s>', model.wv.most_similar('</s>'), '\n')
