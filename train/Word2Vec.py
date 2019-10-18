import time
from config import config
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from gensim.models.word2vec import LineSentence

import os.path
import pandas as pd


def build_dataset(path):
    print(f"Read data from {path}")
    train_data = pd.read_csv(path, encoding='utf-8')
    stoplist = stopword(config.stop_word_path)
    lines = []
    for n in ['Input', 'Report']:
        a = train_data[n].values.tolist()
        b = [str(x).split(' ') for x in a]
        lines.extend(b)
    return lines


def stopword(path):
    word_list = []
    f = open(path, 'r', encoding='utf-8')
    word_list.extend(f.readlines())
    f.close()
    return word_list


def build_word2vec(train_text, save_path):
    print("training word2vec...")
    start_time = time.time()
    # Word2vec+LineSentence训练只用了30s不到，Fasttext要20分钟以上。
    word2vec = Word2Vec(LineSentence(train_text), min_count=5, size=100, workers=5)
    # word2vec = FastText(train_text, sg=1, min_count=5, iter=5, size=200, workers=5)
    print(f'training is done, {time.time() - start_time} seconds elapsed')
    if os.path.exists(config.w2v_bin_path):
        print("already saved")
    else:
        print("start saving...")
        word2vec.save(save_path)
        print("save is ok")
    return word2vec


def build_vocab(wd2vc):
    # The vocab shows in the format of "word: (index, vector)"
    wordvocab = {}
    for i, x in enumerate(wd2vc.wv.index2word):
        wordvocab[x] = wd2vc.wv[x]
    return wordvocab


if __name__ == '__main__':
    # train_texts = build_dataset(config.traindata_path)
    # w2v = FastText.load(config.w2v_bin_path)
    # vocab = build_vocab(w2v)
    # word2vec = build_word2vec(config.corpus_path, config.w2v_bin_path)
    w2v = Word2Vec.load(config.w2v_bin_path)
    assert '汽车' in w2v.wv.vocab
    vocab = build_vocab(w2v)
    print(len(vocab))