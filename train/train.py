import pandas as pd
import numpy as np
from keras.layers import Input, GRU, Dense
from keras.models import Model
from Seq2Seq import Encoder
from config import config
from Word2Vec import build_vocab
from gensim.models.fasttext import FastText


BATCH_SIZE = 2
embedding_dim = 256
units = 1024


def load_dataset(path, num_exaples):
    rawdf = pd.read_csv(path)
    subdf = rawdf.loc[:num_exaples]
    text = [x.split() for x in subdf['Text'].values.tolist()]
    response = [x.split() for x in subdf['Response'].values.tolist()]
    max_input_leng = max((len(x) for x in text))
    max_target_len = max((len(x) for x in response))
    encoder_input_tensor = np.zeros(
        (len(text), max_input_leng, 200)
    )
    decoder_input_tensor = np.zeros(
        (len(response), max_target_len, 200)
    )
    decoder_target_tensor = np.zeros(
        (len(response), max_target_len, 200)
    )
    for i, (input_text, target_text) in enumerate(zip(text, response)):
        for t, word in enumerate(input_text):
            if is_oov(word):
                encoder_input_tensor[i, t, :] = 0
                continue
            encoder_input_tensor[i, t] = vocab[word]
        for t, word in enumerate(target_text):
            if is_oov(word):
                decoder_input_tensor[i, t, :] = 0
                continue
            decoder_input_tensor[i, t] = vocab[word]
            if t > 0:
                if is_oov(word):
                    decoder_target_tensor[i, t-1, :] = 0
                    continue
                decoder_target_tensor[i, t-1] = vocab[word]
    return encoder_input_tensor, decoder_input_tensor, decoder_target_tensor


def is_oov(word):
    return False if word in vocab else True


def sentence2index(sentence, max_len):
    words = sentence.split()
    zeros = [0] * (max_len - len(words))
    indexes = []
    for x in words:
        if is_oov(x):
            index = 0
        else:
            index = vocab[x][0]
        indexes.append(index)
    indexes.extend(zeros)
    return indexes


def build_index(data):
    max_len = max((len(x.split()) for x in data))
    indexes = []
    for x in data:
        index = sentence2index(x, max_len)
        indexes.append(index)
    return indexes, max_len


if __name__ == '__main__':
    # df = pd.read_csv(config.traindata_path)
    w2v = FastText.load(config.w2v_bin_path)
    vocab = build_vocab(w2v)
    num_voc = len(vocab)
    subtext_tensor, subresponse_tensor, subresponse_tftensor = load_dataset(config.traindata_path, 199)
    # subtext_index, max_inputsize = build_index(subtext)
    # subresponse_index, max_inputsize2 = build_index(subresponse)
    # print(subtext_index[:2], subresponse_index[:2])
    # encoder = Encoder(max_inputsize, embedding_dim, units, BATCH_SIZE)
    encoder = GRU(1024, return_sequences=True, return_state=True, batch_size=BATCH_SIZE)
    encoder_inputs = Input(shape=(None, 200))
    sample_output, sample_hidden = encoder(encoder_inputs)
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))
    decoder_inputs = Input(shape=(None, 200))
    decoder = GRU(1024, return_state=True, return_sequences=True, batch_size=BATCH_SIZE)
    decoder_outputs, _ = decoder(decoder_inputs, initial_state=sample_hidden)
    decoder_dense = Dense(200, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit([subtext_tensor, subresponse_tensor], subresponse_tftensor,
              batch_size=BATCH_SIZE,
              epochs=10,
              validation_split=0.2)






