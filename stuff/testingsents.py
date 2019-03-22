import re
import gensim
import logging
import multiprocessing
import os
import sys
import pickle
import random
import threading
from threading import Thread
from time import time
raw_html = "<s> The John this is a test of Named Entities Parsing it works with Middle or End Items </s>"


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext


def fixlist(cleantext):
    entities = cleantext.split()
    text = cleantext.split()
    for i in range(1, len(text) - 3):
        new = text[i] + ' ' + text[i + 1] + ' ' + text[i + 2]
        entities += text[:i]
        entities.append(new)
        entities += text[i + 3:]
    return entities


lines = open(
    '/Users/vincent/Desktop/mounted/enwiki/AB/wiki_00',
    encoding='utf-8',
    errors='ignore')


class SentenceWorker(Thread):
    def __init__(self, queue):
        self.queue = queue

    def run(self):
        while True:
            sentence, n = self.queue.get()


class Test(object):
    def __init__(self, lines):
        self.lines = lines

    def __iter__(self):
        for line in self.lines:
            line = cleanhtml(line)
            line = line.strip()
            if line == '':
                continue
            list_of_sentences = tokenize(line)
            # print(list_of_sentences)
            # this is a list of sentences and seperates punctuation from a word with a space
            # 'Hello, world' -> 'Hello , world'
            paragraph = ' '.join(list_of_sentences)
            list_of_words = [
                word for word in paragraph.split() if word.isalpha()
            ]

            yield list_of_words


# TODO replace this with a function that creates an entity list
# print(list_of_words)

# ['After World War II the number of inmates in prison camps and colonies , again , rose sharply , reaching approximately 2.5 million people by the early 1950s ( about 1.7 million of whom were in camps ) .']
# ['After', 'World', 'War', 'II', 'the', 'number', 'of', 'inmates', 'in', 'prison', 'camps', 'and', 'colonies', 'again', 'rose', 'sharply', 'reaching', 'approximately', 'million', 'people', 'by', 'the', 'early', 'about', 'million', 'of', 'whom', 'were', 'in', 'camps']