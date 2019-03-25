import tensorflow as tf
import pandas as pd
import numpy as np

import re
import time
import gc

from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from nltk.corpus import stopwords
from data_prep_tf import *

codes = ["<UNK>", "<PAD>", "<EOS>", "<SOS>"]

data = pd.read_csv('/mounted/data/test.csv')

embeddings_index = {}
with open('/mounted/data/numberbatch.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding
    del f

gc.collect()

print('Word embeddings:', len(embeddings_index))

missing_words = 0
threshold = 10

cleaned_words, clean_definitions = get_data(data)

word_counts = get_counts(cleaned_words, clean_definitions)
for word, count in word_counts.items():
    if count > threshold:
        if word not in embeddings_index:
            missing_words += 1

missing_ratio = round(missing_words / len(word_counts), 4) * 100

print("Number of words missing from CN:", missing_words)
print("Percent of words that are missing from vocabulary: {}%".format(
    missing_ratio))

vocab_to_int = {}

value = 0
for word, count in word_counts.items():
    if count >= threshold or word in embeddings_index:
        vocab_to_int[word] = value
        value += 1

# Add tokens:
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)

vocab_to_int = {}

value = 0
for word, count in word_counts.items():
    if count >= threshold or word in embeddings_index:
        vocab_to_int[word] = value
        value += 1

# Special tokens that will be added to our vocab
codes = ["<UNK>", "<PAD>", "<EOS>", "<SOS>"]

# Add codes to vocab
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)

# Dictionary to convert integers to words
int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word

usage_ratio = round(len(vocab_to_int) / len(word_counts), 4) * 100

print("Total number of unique words:", len(word_counts))
print("Number of words we will use:", len(vocab_to_int))
print("Percent of words we will use: {}%".format(usage_ratio))
embedding_dim = 300
nb_words = len(vocab_to_int)

word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
for word, i in vocab_to_int.items():
    if word in embeddings_index:
        word_embedding_matrix[i] = embeddings_index[word]
    else:
        # If word not in CN, create a random embedding for it
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        embeddings_index[word] = new_embedding
        word_embedding_matrix[i] = new_embedding

# Check if value matches len(vocab_to_int)
print(len(word_embedding_matrix))


def convert_to_ints(word, word_count, unk_count, eos=False):
    '''Convert words in word to an integer.
       If word is not in vocab_to_int, use UNK's integer.
       Total the number of words and UNKs.
       Add EOS token to the end of words'''
    ints = []
    for sentence in word:
        sentence_ints = []
        for word in sentence.split():
            word_count += 1
            if word in vocab_to_int:
                sentence_ints.append(vocab_to_int[word])
            else:
                sentence_ints.append(vocab_to_int["<UNK>"])
                unk_count += 1
        if eos:
            sentence_ints.append(vocab_to_int["<EOS>"])
        ints.append(sentence_ints)
    return ints, word_count, unk_count


word_count = 0
unk_count = 0

int_words, word_count, unk_count = convert_to_ints(cleaned_words, word_count,
                                                   unk_count)
int_definitions, word_count, unk_count = convert_to_ints(
    clean_definitions, word_count, unk_count, eos=True)

unk_percent = round(unk_count / word_count, 4) * 100

print("Total number of words in words:", word_count)
print("Total number of UNKs in words:", unk_count)
print("Percent of words that are UNK: {}%".format(unk_percent))


def create_lengths(word):
    '''Create a data frame of the sentence lengths from a word'''
    lengths = []
    for sentence in word:
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])


lengths_words = create_lengths(int_words)
lengths_definitions = create_lengths(int_definitions)

print("Words:")
print(lengths_words.describe())
print()
print("Definitions:")
print(lengths_definitions.describe())

print(np.percentile(lengths_words.counts, 90))
print(np.percentile(lengths_words.counts, 95))
print(np.percentile(lengths_words.counts, 99))

print(np.percentile(lengths_definitions.counts, 90))
print(np.percentile(lengths_definitions.counts, 95))
print(np.percentile(lengths_definitions.counts, 99))


def unk_counter(sentence):
    '''Counts the number of time UNK appears in a sentence.'''
    unk_count = 0
    for word in sentence:
        if word == vocab_to_int["<UNK>"]:
            unk_count += 1
    return unk_count


sorted_words = []
sorted_definitions = []
max_definition_length = 84
max_word_length = 13
min_length = 2
unk_word_limit = 3
unk_definition_limit = 2

for length in range(min(lengths_words.counts), max_word_length):
    for count, words in enumerate(int_definitions):
        if (len(int_definitions[count]) >= min_length
                and len(int_definitions[count]) <= max_definition_length
                and len(int_words[count]) >= min_length
                and unk_counter(int_definitions[count]) <= unk_definition_limit
                and unk_counter(int_words[count]) <= unk_word_limit
                and length == len(int_words[count])):
            sorted_definitions.append(int_definitions[count])
            sorted_words.append(int_words[count])

# Compare lengths to ensure they match
print(len(sorted_definitions))
print(len(sorted_words))

# Tensorflow Code:


def model_inputs():
    '''Create palceholders for inputs to the model'''

    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    definition_length = tf.placeholder(
        tf.int32, (None, ), name='definition_length')
    max_definition_length = tf.reduce_max(
        definition_length, name='max_dec_len')
    word_length = tf.placeholder(tf.int32, (None, ), name='word_length')

    return input_data, targets, lr, keep_prob, definition_length, max_definition_length, word_length


def process_encoding_input(target_data, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <SOS> to the begining of each batch'''

    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat(
        [tf.fill([batch_size, 1], vocab_to_int['<SOS>']), ending], 1)

    return dec_input


def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs,
                   keep_prob):
    '''Create the encoding layer'''

    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf.contrib.rnn.LSTMCell(
                rnn_size,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(
                cell_fw, input_keep_prob=keep_prob)

            cell_bw = tf.contrib.rnn.LSTMCell(
                rnn_size,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(
                cell_bw, input_keep_prob=keep_prob)

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                rnn_inputs,
                sequence_length,
                dtype=tf.float32)
    # Join outputs since we are using a bidirectional RNN
    enc_output = tf.concat(enc_output, 2)

    return enc_output, enc_state


def training_decoding_layer(dec_embed_input, definition_length, dec_cell,
                            initial_state, output_layer, vocab_size,
                            max_definition_length):
    '''Create the training logits'''

    training_helper = tf.contrib.seq2seq.TrainingHelper(
        inputs=dec_embed_input,
        sequence_length=definition_length,
        time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(
        dec_cell, training_helper, initial_state, output_layer)

    training_logits, _, __ = tf.contrib.seq2seq.dynamic_decode(
        training_decoder,
        output_time_major=False,
        impute_finished=True,
        maximum_iterations=max_definition_length)
    return training_logits


def inference_decoding_layer(embeddings, start_token, end_token, dec_cell,
                             initial_state, output_layer,
                             max_definition_length, batch_size):
    '''Create the inference logits'''

    start_tokens = tf.tile(
        tf.constant([start_token], dtype=tf.int32), [batch_size],
        name='start_tokens')

    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        embeddings, start_tokens, end_token)

    inference_decoder = tf.contrib.seq2seq.BasicDecoder(
        dec_cell, inference_helper, initial_state, output_layer)

    inference_logits, _, __ = tf.contrib.seq2seq.dynamic_decode(
        inference_decoder,
        output_time_major=False,
        impute_finished=True,
        maximum_iterations=max_definition_length)

    return inference_logits


def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state,
                   vocab_size, word_length, definition_length,
                   max_definition_length, rnn_size, vocab_to_int, keep_prob,
                   batch_size, num_layers):
    '''Create the decoding cell and attention for the training and inference decoding layers'''

    for layer in range(num_layers):
        with tf.variable_scope('decoder_{}'.format(layer)):
            lstm = tf.contrib.rnn.LSTMCell(
                rnn_size,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            dec_cell = tf.contrib.rnn.DropoutWrapper(
                lstm, input_keep_prob=keep_prob)

    output_layer = Dense(
        vocab_size,
        kernel_initializer=tf.truncated_normal_initializer(
            mean=0.0, stddev=0.1))

    attn_mech = tf.contrib.seq2seq.BahdanauAttention(
        rnn_size,
        enc_output,
        word_length,
        normalize=False,
        name='BahdanauAttention')

    dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attn_mech,
                                                   rnn_size)
    dec_cell.zero_state(
        batch_size=batch_size, dtype=tf.float32).clone(cell_state=enc_state)

    initial_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(
        enc_state[0], _zero_state_tensors(rnn_size, batch_size, tf.float32))
    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(
            dec_embed_input, definition_length, dec_cell, initial_state,
            output_layer, vocab_size, max_definition_length)
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(
            embeddings, vocab_to_int['<SOS>'], vocab_to_int['<EOS>'], dec_cell,
            initial_state, output_layer, max_definition_length, batch_size)

    return training_logits, inference_logits


def seq2seq_model(input_data, target_data, keep_prob, word_length,
                  definition_length, max_definition_length, vocab_size,
                  rnn_size, num_layers, vocab_to_int, batch_size):
    '''Use the previous functions to create the training and inference logits'''

    # Use Numberbatch's embeddings and the newly created ones as our embeddings
    embeddings = word_embedding_matrix

    enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
    enc_output, enc_state = encoding_layer(rnn_size, word_length, num_layers,
                                           enc_embed_input, keep_prob)

    dec_input = process_encoding_input(target_data, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)

    training_logits, inference_logits = decoding_layer(
        dec_embed_input, embeddings, enc_output, enc_state, vocab_size,
        word_length, definition_length, max_definition_length, rnn_size,
        vocab_to_int, keep_prob, batch_size, num_layers)

    return training_logits, inference_logits


def pad_sentence_batch(sentence_batch):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [
        sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence))
        for sentence in sentence_batch
    ]


def get_batches(definitions, words, batch_size):
    """Batch definitions, words, and the lengths of their sentences together"""
    for batch_i in range(0, len(words) // batch_size):
        start_i = batch_i * batch_size
        definitions_batch = definitions[start_i:start_i + batch_size]
        words_batch = words[start_i:start_i + batch_size]
        pad_definitions_batch = np.array(pad_sentence_batch(definitions_batch))
        pad_words_batch = np.array(pad_sentence_batch(words_batch))

        # Need the lengths for the _lengths parameters
        pad_definitions_lengths = []
        for definition in pad_definitions_batch:
            pad_definitions_lengths.append(len(definition))

        pad_words_lengths = []
        for word in pad_words_batch:
            pad_words_lengths.append(len(word))

        yield pad_definitions_batch, pad_words_batch, pad_definitions_lengths, pad_words_lengths


epochs = 100
batch_size = 64
rnn_size = 256
num_layers = 2
learning_rate = 0.005
keep_probability = 0.75

train_graph = tf.Graph()
# Set the graph to default to ensure that it is ready for training
with train_graph.as_default():

    # Load the model inputs
    input_data, targets, lr, keep_prob, definition_length, max_definition_length, word_length = model_inputs(
    )

    # Create the training and inference logits
    training_logits, inference_logits = seq2seq_model(
        tf.reverse(input_data, [-1]), targets, keep_prob, word_length,
        definition_length, max_definition_length,
        len(vocab_to_int) + 1, rnn_size, num_layers, vocab_to_int, batch_size)

    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_logits.rnn_output, 'logits')
    inference_logits = tf.identity(
        inference_logits.sample_id, name='predictions')

    # Create the weights for sequence_loss
    masks = tf.sequence_mask(
        definition_length,
        max_definition_length,
        dtype=tf.float32,
        name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets,
                                                masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var)
                            for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
print("Graph is built.")

start = 200000
end = start + 50000
sorted_definitions_short = sorted_definitions[start:end]
sorted_words_short = sorted_words[start:end]
print("The shortest word length:", len(sorted_words_short[0]))
print("The longest word length:", len(sorted_words_short[-1]))

learning_rate_decay = 0.95
min_learning_rate = 0.0005
display_step = 20  # Check training loss after every 20 batches
stop_early = 0
stop = 3  # If the update loss does not decrease in 3 consecutive update checks, stop training
per_epoch = 3  # Make 3 update checks per epoch
update_check = (len(sorted_words_short) // batch_size // per_epoch) - 1

update_loss = 0
batch_loss = 0
definition_update_loss = [
]  # Record the update losses for saving improvements in the model

checkpoint = "best_model.ckpt"
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    # If we want to continue training a previous session
    #loader = tf.train.import_meta_graph("./" + checkpoint + '.meta')
    #loader.restore(sess, checkpoint)

    for epoch_i in range(1, epochs + 1):
        update_loss = 0
        batch_loss = 0
        for batch_i, (definitions_batch, words_batch, definitions_lengths,
                      words_lengths) in enumerate(
                          get_batches(sorted_definitions_short,
                                      sorted_words_short, batch_size)):
            start_time = time.time()
            _, loss = sess.run(
                [train_op, cost], {
                    input_data: words_batch,
                    targets: definitions_batch,
                    lr: learning_rate,
                    definition_length: definitions_lengths,
                    word_length: words_lengths,
                    keep_prob: keep_probability
                })

            batch_loss += loss
            update_loss += loss
            end_time = time.time()
            batch_time = end_time - start_time

            if batch_i % display_step == 0 and batch_i > 0:
                print(
                    'Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                    .format(epoch_i, epochs, batch_i,
                            len(sorted_words_short) // batch_size,
                            batch_loss / display_step,
                            batch_time * display_step))
                batch_loss = 0

            if batch_i % update_check == 0 and batch_i > 0:
                print("Average loss for this update:",
                      round(update_loss / update_check, 3))
                definition_update_loss.append(update_loss)

                # If the update loss is at a new minimum, save the model
                if update_loss <= min(definition_update_loss):
                    print('New Record!')
                    stop_early = 0
                    saver = tf.train.Saver()
                    saver.save(sess, checkpoint)

                else:
                    print("No Improvement.")
                    stop_early += 1
                    if stop_early == stop:
                        break
                update_loss = 0

        # Reduce learning rate, but not below its minimum value
        learning_rate *= learning_rate_decay
        if learning_rate < min_learning_rate:
            learning_rate = min_learning_rate

        if stop_early == stop:
            print("Stopping Training.")
            break


def word_to_seq(word):
    '''Prepare the word for the model'''

    word = clean_word(word)
    return [
        vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in word.split()
    ]


random = np.random.randint(0, len(cleaned_words))
input_sentence = cleaned_words[random]
word = word_to_seq(cleaned_words[random])

checkpoint = "./best_model.ckpt"

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    word_length = loaded_graph.get_tensor_by_name('word_length:0')
    definition_length = loaded_graph.get_tensor_by_name('definition_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    #Multiply by batch_size to match the model's input parameters
    answer_logits = sess.run(
        logits, {
            input_data: [word] * batch_size,
            definition_length: [np.random.randint(5, 8)],
            word_length: [len(word)] * batch_size,
            keep_prob: 1.0
        })[0]

# Remove the padding from the tweet
pad = vocab_to_int["<PAD>"]

print('Original word:', input_sentence)

print('\nword')
print('  Word Ids:    {}'.format([i for i in word]))
print('  Input Words: {}'.format(" ".join([int_to_vocab[i] for i in word])))

print('\ndefinition')
print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format(" ".join(
    [int_to_vocab[i] for i in answer_logits if i != pad])))
