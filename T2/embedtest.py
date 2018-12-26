import api_client
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
import pickle
import sys
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint


# Model variables
n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 25
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

def create_vocabulory(data, size):
    t = Tokenizer(lower = True)
    all_questions = []

    for i in data:
        all_questions.append(i['question1'])
        all_questions.append(i['question2'])
    
    debug("All questions")
    debug(len(all_questions))
    
    stop_words = set(stopwords.words('english'))

    t.fit_on_texts(all_questions)
    vocab_temp = t.word_index

    words = list(vocab_temp.keys())

    for i in words:
        if(i in stop_words):
            del vocab_temp[i]

    i = 0
    words = list(vocab_temp.keys()) 
    vocab = {}
    while(i < size):
        try:
            vocab[words[i]] = i + 1
        except:
            break
        i += 1

    debug("vocabulory size")
    debug(len(vocab))

    return vocab                                

if __name__ == "__main__":
    data_size = int(sys.argv[1])
    vocab_size = int(sys.argv[2])

    try:
        data = load_list_from_pickle(data_size)
    except:
        data = getData(data_size)
    
    debug("Dataset")
    debug(data[:3])

    vocabulary = create_vocabulory(data, vocab_size)

    embedding_matrix = create_embedding_matrix(vocabulary)

    debug(vocabulary)    
    debug(embedding_matrix)                     

def create_model():
    def exponent_neg_manhattan_distance(left, right):
        ''' Helper function for the similarity estimate of the LSTMs outputs'''
        return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

    # The visible layer
    left_input = Input(shape=(max_seq_length,), dtype='int32')
    right_input = Input(shape=(max_seq_length,), dtype='int32')

    embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=False)

    # Embedded version of the inputs
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)

    # Since this is a siamese network, both sides share the same LSTM
    shared_lstm = LSTM(n_hidden)

    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)

    # Calculates the distance as defined by the MaLSTM model
    malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

    # Pack it all up into a model
    malstm = Model([left_input, right_input], [malstm_distance])

    # Adadelta optimizer, with gradient clipping by norm
    optimizer = Adadelta(clipnorm=gradient_clipping_norm)

    malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    # Start training
    training_start_time = time()

    malstm_trained = malstm.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, nb_epoch=n_epoch,
                                validation_data=([X_validation['left'], X_validation['right']], Y_validation))

    print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))
  