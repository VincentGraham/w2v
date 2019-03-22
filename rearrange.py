from multiprocessing import Process, Queue, Pool
import timeit
import os
import random
import logging
import re
import queue

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

compiled_expression = re.compile('<.*?>')


def cleanhtml(raw_html):
    global compiled_expression
    cleantext = re.sub(compiled_expression, ' ', raw_html)
    return cleantext


def cleanStrings(inStr):
    a = inStr.find('<')
    b = inStr.find('>')
    if a < 0 and b < 0:
        return inStr
    return cleanString(inStr[a:b - a])


def tokenize(paragraph):
    """
    Paragraph is a paragraph of plain text.
    Returns a list of sentences with punctuation split from words by a space.
    """
    out = []
    if isinstance(paragraph, str):
        for s in re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s',
                          paragraph):
            if s[-1].isdigit() or s[-1].isalpha():
                s = s.replace(',', '')
                out.append(s)
            else:
                s = s.replace(',', '')
                out.append(s[:-1] + " .")
    # if (len(out) > 2):
    return out


def check_case(sentence):
    l = len(sentence)
    if l < 4:
        return sentence
    entities = []
    out = sentence[:]
    sentence.append(' ')
    for idx, word in enumerate(sentence):
        if idx + 1 == l:
            pass
        elif word[0].isupper():
            i = 1
            entity = word
            if idx < l - 2:
                while (sentence[idx + i][0].isupper()
                       or sentence[idx + i + 1][0].isupper()):
                    entity += ' ' + sentence[idx + i]
                    if sentence[idx + i + 1][0].isupper():
                        entity += ' ' + sentence[idx + i + 1]
                        i += 2
                    else:
                        i += 1
                    if idx + i + 1 >= l:
                        break
            elif idx < l:  #s[24] is the last in l=25
                while (sentence[idx + i][0].isupper()):
                    entity += ' ' + sentence[idx + i]
                    i += 1
            entities.append((idx, entity))

    # for idx, e in entities:
    # if len(e.split(' ')) > 1:
    #     out += ['</s>', '<s>'] + sentence[:idx] + [
    #         e
    #     ] + sentence[idx + len(e.split(' ')):-1] + ['</s>']
    #     # if idx == 23:
    #     #     print(out, entity, idx, sentence)
    #     #     break
    if len(entities) > 1:
        e = max(entities, key=len)
        out += ['</s>', '<s>'] + sentence[:e[0]] + [
            e[1]
        ] + sentence[e[0] + len(e[1].split()):-1] + ['</s>']
    out = ['<s>'] + [x for x in out
                     if x != ' '] + ['</s>'] if out[-1] != ['</s>'] else []
    return out


def get_sentences(file_path):
    start_time = timeit.default_timer()
    sentences = []
    for line in open(file_path, encoding='utf-8', errors='ignore'):
        line = cleanhtml(line)
        line = line.strip()
        if line == '':
            continue
        list_of_sentences = tokenize(line)
        # print(list_of_sentences)
        # this is a list of sentences and seperates punctuation from a word with a space
        # 'Hello, world' -> 'Hello , world'

        for sentence in list_of_sentences:
            while (len(sentence.split()) < 5):
                sentence += ' <p>'
            if random.random() < 0.02:
                to_replace = sentence.split()
                to_replace[int(
                    random.random() * (len(to_replace) - 1))] = '<u>'
                toks = to_replace
                sentence = ' '.join(toks)

            list_of_words = [
                word for word in sentence.split()
                if ((word.isalpha() or word.isdigit()) or word == '<p>'
                    or word == '<u>' or '-' in word)
            ]
            result = check_case(list_of_words)
            sentences += result
    logging.info('Done at {} in {:10.3f}s'.format(
        file_path,
        timeit.default_timer() - start_time))
    start_time = timeit.default_timer()
    return sentences


def main(tasks, outqueue):
    # while not tasks.empty():
    #     task = tasks.get()
    #     x = get_sentences(task)
    #     outqueue.put(x)
    while True:
        try:
            a = tasks.get(timeout=1)
        except queue.Empty:
            return
        x = get_sentences(a)
        outqueue.put(x)


# start_time = timeit.default_timer()
# for root, dirs, files in os.walk('/Users/vincent/Desktop/mounted/enwiki/AC'):
#     paths = []
#     for filename in files:
#         file_path = root + '/' + filename
#         if filename == '.DS_Store':
#             continue
#         paths.append(file_path)
#     pool = Pool(12)
#     y = pool.map(get_sentences, paths)
#     print('Done at {}'.format(root))
#     print(timeit.default_timer() - start_time)
# print(timeit.default_timer() - start_time)

# On 100MB (dir AC, process after assembling 1 big list 100mb)
# 1 Pool  : 18.69576609700016
# 2 Pool  : 13.50921916900188
# 4 Pool  : 11.858966263000184
# 24 Pool : 13.905261799998698

# On 100MB, process per file
# 1 Pool  :           93.96200378099820 ->
# 4 Pool  :           91.64671368599738 -> 32.85915138499695
# 6 Pool  :           95.65391662299953 -> 25.084024065996346
# 12Pool  :                             -> 18.679853446003108
# Only loading files: 81.75308772099743
