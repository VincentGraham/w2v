import gensim
import pickle
import os
import csv
import pandas as pd

from rearrange import cleanhtml, tokenize, check_case


def save_word2vec(model=None):
    if model is None:
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
            'data/lowercase_model_pos.txt')
        weights_matrix = torch.FloatTensor(word2vec_model.vectors)
        word2vec_model.save_word2vec_format('data/word2vec.bin', binary=True)
        print("loaded word2vec layer")

    else:
        word2vec_model = load_word2vec_model()

    with open("data/vocab.pkl", 'wb') as file:
        pickle.dump(list(word2vec_model.vocab.keys()), file)

    file.close()


def load_word2vec_vocab():
    with open("data/vocab.pkl", "rb") as file:
        vocab_list = pickle.load(file)
    file.close()
    return vocab_list


def load_word2vec_model():
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
        'data/model.bin', binary=True)
    print("Converting Model to weight vectors...")
    return word2vec_model


def alt_save():
    w = load_word2vec_model()
    w.save_word2vec_format('data/model.txt', binary=False)


# alt_save()

# save_word2vec(load_word2vec_model())


def get_sentences_csv():
    sentences = []
    flies = []
    out = []
    idx = 0
    for root, dirs, files in os.walk('/Users/vincent/Desktop/mounted/enwiki'):
        for file_name in files:
            if file_name == ".DS_Store":
                continue
            file_name = root + '/' + file_name
            flies.append(file_name)

    data = [[] for _ in range(len(flies) * 2)]

    for file_path in flies:
        i = 0
        for line in open(file_path, encoding='utf-8', errors='ignore'):
            line = cleanhtml(line)
            line = line.strip()
            if line == '':
                continue
            # print(list_of_sentences)
            # this is a list of sentences and seperates punctuation from a word with a space
            # 'Hello, world' -> 'Hello , world'
            if i < 1:
                i += 1
                data[idx] = [str(line), str(line)]
            else:
                data[idx][1] = str(line)
                idx += 1
                i += 1
            if i >= 2:
                break
    return data


def make():
    with open('data/first.csv', 'w+') as csvfile:
        fieldnames = ['sentence', 'article']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        i = -1
        for item in get_sentences_csv():
            if item == []:
                continue
            if item[0] != "" and item[1] != "":
                writer.writerow({'sentence': item[0], 'article': item[1]})
    fi = open('data/first.csv', 'rb')
    data = fi.read()
    fi.close()
    fo = open('data/test.csv', 'wb')
    fo.write(data.replace(b'\x00', ''.encode(encoding='UTF-8')))
    fo.close()


# make()


def load_csv_for_tensor():
    """Call datadict[sentence][i] and datadict[article][i] with same i"""
    data = pd.read_csv("data/test.csv")
    data_dict = {col: data[col] for col in data.columns}
    return data_dict
