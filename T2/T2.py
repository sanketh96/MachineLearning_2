import api_client
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
import pickle
import sys
import numpy as np
import itertools
import datetime
import json
from time import time
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization

# Model variables
n_hidden = 50
num_dense = 25
gradient_clipping_norm = 1.25
# batch_size = 64
# n_epoch = 25
EMBEDDING_DIM = 100
MAX_NB_WORDS = 5000
DEBUG = True


def debug(*args):
    if(DEBUG):
        print(*args)

def store_list_as_pickle(save_list, sample_size):
    name = "kaggle_data_" + str(sample_size)+".pickle"
    with open(name, 'wb') as f:
        pickle.dump(save_list, f)

def load_list_from_pickle(sample_size):
    filename = "kaggle_data_" + str(sample_size)+".pickle" 
    with open(filename, 'rb') as f:
        mynewlist = pickle.load(f)
    return mynewlist

def getData(size):
    client = api_client.ApiClient("eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE1NTQ2MjcxMDMsImlhdCI6MTUzOTA3NTEwMywibmJmIjoxNTM5MDc1MTAzLCJpZGVudGl0eSI6OX0.I4e6eJThErY_nznQUAJqOWKYnY0Z46WquZnEHX3ygck")
    global_data = []
    iterations = size // 10000
    if size > 10000:
        new_size = 10000
        for i in range(iterations):
            data = client.get_kaggle_quora_data(num_samples = new_size)
            global_data.extend(data)
    data = client.get_kaggle_quora_data(num_samples = (size % 10000))
    global_data.extend(data)
    
    debug("no of records obtained : ", len(global_data))
    # flatten the list
    # flat_list = []
    # for sublist in global_data:
    #     for item in sublist:
    #         flat_list.append(item)
    #store the pickle
    store_list_as_pickle(global_data, size)
    return global_data

def getData2(size):
    with open('dataset.json') as f:
        dataset = json.load(f)
    
    return dataset[:size]

def create_embedding_matrix(vocab,file_path='glove.6B.100d.txt'):
    embeddings_index = {}
    f = open(file_path,encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    # prepare embedding matrix
    num_words = min(MAX_NB_WORDS, len(vocab) + 1)
    
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

    for word, i in vocab.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros
            embedding_matrix[i] = embedding_vector 
    return embedding_matrix

def get_maxlen(sequences):
    return len(max(sequences, key = lambda x: len(x)))

def pre_process_dataset(data, size, split_ratio):
    global pad_length

    t = Tokenizer(lower = True)
    all_questions = []
    stop_words = set(stopwords.words('english'))
    question1 = []
    question2 = []
    labels = []
    for i in data:
        q1 = i['question1']
        q2 = i['question2']
        q1 = ' '.join([word for word in q1.split() if word not in stop_words])
        q2 = ' '.join([word for word in q2.split() if word not in stop_words])
        question1.append(q1)
        question2.append(q2)

        labels.append(int(i['is_duplicate']))

    debug("All questions")
    debug(len(all_questions))
    
    all_questions = question1 + question2

    t.fit_on_texts(all_questions)
    vocab_temp = t.word_index
    # debug("vocab_temp: ", vocab_temp)
    i = 0
    words = list(vocab_temp.keys()) 
    vocab = {}
    while(i < size):
        vocab[words[i]] = vocab_temp[words[i]]
        i += 1

    debug("vocabulory size")
    debug(len(vocab))

    pad_length = get_maxlen(all_questions)
    debug("pad_length = ", pad_length)

    question1_sequences = np.array(pad_sequences(t.texts_to_sequences(question1), maxlen = pad_length))
    question2_sequences = np.array(pad_sequences(t.texts_to_sequences(question2), maxlen = pad_length))

    temp = np.arange(len(data))
    np.random.shuffle(temp)

    test_data_size = int(split_ratio * len(data))


    question1_train = question1_sequences[temp[:(len(data) - test_data_size)]]
    question1_test = question1_sequences[temp[(len(data) - test_data_size):]]
    question2_train = question2_sequences[temp[:(len(data) - test_data_size)]]
    question2_test = question2_sequences[temp[(len(data) - test_data_size):]]
    labels = np.array(labels)
    labels_train = labels[temp[:(len(data) - test_data_size)]]
    labels_test = labels[temp[(len(data) - test_data_size):]]

    debug("DEBUG")
    debug(question1[temp[500]])
    debug(question2[temp[500]])
    debug(labels[500])

    return vocab, question1_train, question2_train, question1_test, question2_test, labels_train, labels_test


def create_model(embedding_matrix):
    max_seq_length = pad_length
    def exponent_neg_manhattan_distance(left, right):
        ''' Helper function for the similarity estimate of the LSTMs outputs'''
        return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

    # The visible layer
    left_input = Input(shape=(max_seq_length,), dtype='int32')
    right_input = Input(shape=(max_seq_length,), dtype='int32')

    embedding_layer = Embedding(len(embedding_matrix),EMBEDDING_DIM, weights=[embedding_matrix], input_length=max_seq_length, trainable=False)

    # Embedded version of the inputs
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)

    # Since this is a siamese network, both sides share the same LSTM
    shared_lstm = LSTM(n_hidden)

    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)

    #conactenated output
    merged = concatenate([left_output,right_output])
    merged = Dropout(rate = 0.2)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation='elu')(merged)
    merged = Dropout(rate=0.2)(merged)
    merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)


    # Calculates the distance as defined by the MaLSTM model
    malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

    # Pack it all up into a model
    malstm = Model([left_input, right_input], [preds])

    # Adadelta optimizer, with gradient clipping by norm
    
    return malstm

def train_model(malstm, X_train_q1, X_train_q2, Y_train, batch_size = 16, n_epoch = 5):
    optimizer = Adadelta(clipnorm=gradient_clipping_norm)
    malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    # Start training
    training_start_time = time()
    malstm_trained = malstm.fit([X_train_q1, X_train_q2], Y_train, batch_size=batch_size, nb_epoch = n_epoch, validation_split = 0.2, verbose = 2)
    print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))
    return malstm_trained

if __name__ == "__main__":
    data_size = int(sys.argv[1])
    vocab_size = int(sys.argv[2])

    # try:
    #     data = load_list_from_pickle(data_size)
    #     debug("read from pickle")
    # except:
    #     data = getData(data_size)
    
    data = getData2(data_size)

    debug("Dataset")
    debug(data[:3])

    vocabulary, question1_train, question2_train, question1_test, question2_test, labels_train, labels_test = pre_process_dataset(data, vocab_size, split_ratio = 0.3)

    # debug("Debuging train-test split!")
    # debug(question1_train[0])
    # debug(question2_train[0])
    # debug(labels_train[0])

    embedding_matrix = create_embedding_matrix(vocabulary)

    #debug(vocabulary)
    #debug(embedding_matrix)

    model = create_model(embedding_matrix)
    trained_model = train_model(model, question1_train, question2_train, labels_train)