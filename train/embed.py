import pandas as pd
import numpy as np
from config import config
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec

max_features = 300
maxlen = 300
embed_size = 100
max_length_inp, max_length_targ = 500, 50


def tokenize(lang, max_len):
    # 这里max_features是要挑选最常用的词汇
    tokenizer = Tokenizer(filters='', lower=False, num_words=max_features)
    tokenizer.fit_on_texts(lang)
    # 114k 的词典
    # 注意：keras API的word_index是从1开始的，0要预留出来，因为进行了补零操作
    word_index = tokenizer.word_index
    tensor = tokenizer.texts_to_sequences(lang)
    tensor = pad_sequences(tensor, padding='post', maxlen=max_len)
    return tensor, tokenizer, word_index


def load_train(num_samples=None):
    data = pd.read_csv(config.traindata_path)
    if num_samples is None:
        source = [str(m) for m in data['Input'].values.tolist()]
        target = [str(m) for m in data['Report'].values.tolist()]
    else:
        source = data['Input'].values.tolist()
        target = data['Report'].values.tolist()
        source = [str(m) for m in source[:num_samples]]
        target = [str(m) for m in target[:num_samples]]
    input_tensor, tokenizer1, word_index1 = tokenize(source, max_length_inp)
    target_tensor, tokenizer2, word_index2 = tokenize(target, max_length_targ)
    return input_tensor, target_tensor, word_index1, word_index2, tokenizer1, tokenizer2


def get_embedding(model):
    input_tensor, target_tensor, word_index1, word_index2, tokenizer1, tokenizer2 = load_train()
    # encoder embedding
    word_size1 = min(max_features, len(word_index1))
    embedding_matrix1 = np.zeros((word_size1, embed_size))
    # 此处的i是从1开始的
    for word, i in word_index1.items():
        if i >= word_size1: break
        if word not in model.wv.vocab:
            embedding_matrix1[i] = np.random.uniform(low=-0.025, high=0.025, size=embed_size)
        else:
            embedding_matrix1[i] = model.wv[word]
    # decoder embedding
    word_size2 = min(max_features, len(word_index2))
    embedding_matrix2 = np.zeros((word_size2, embed_size))
    for word, i in word_index2.items():
        if i >= word_size2: break
        if word not in model.wv.vocab:
            embedding_matrix2[i] = np.random.uniform(low=-0.025, high=0.025, size=embed_size)
        else:
            embedding_matrix2[i] = model.wv[word]
    return embedding_matrix1, embedding_matrix2


if __name__ == "__main__":
    w2v_model = Word2Vec.load(config.w2v_bin_path)
    get_embedding(w2v_model)