from config import config
import jieba
import pandas as pd
import re


def cut(sentence):
    # jieba切词，默认精确模式，全模式cut参数：cut_all=True
    return ' '.join(jieba.cut(sentence))


def stopword(path):
    word_list = []
    f = open(path, 'r', encoding='utf-8')
    word_list.extend(f.readlines())
    f.close()
    return word_list


def tokenizer(path):
    raw_train_data = pd.read_csv(path, encoding='utf-8')
    raw_train_data = raw_train_data.fillna("")
    stoplist = stopword(config.stop_word_path)
    for n in ['Question', 'Dialogue', 'Report']:
        for i, m in enumerate(raw_train_data[n]):
            pattern = re.compile(r'[\w|\d]+')
            words = pattern.findall(m)
            words = [w for w in words if w not in stoplist]
            words = ''.join(words)
            raw_train_data[n][i] = cut(words)
    raw_train_data.to_csv(config.train_seg_path)
    return


tokenizer(config.train_path)
