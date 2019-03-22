import gensim
import pickle


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


# save_word2vec(load_word2vec_model())
