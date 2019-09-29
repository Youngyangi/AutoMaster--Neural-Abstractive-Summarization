from config import config
import jieba
import pandas as pd
import re


def cut(sentence):
    # jieba切词，默认精确模式，全模式cut参数：cut_all=True
    return ' '.join(jieba.cut(sentence))


if __name__ == '__main__':
    raw_train_data = pd.read_csv(config.train_path, encoding='utf-8')
    raw_train_data = raw_train_data.fillna("")
    for n in ['Question', 'Dialogue', 'Report']:
        for i, m in enumerate(raw_train_data[n]):
            pattern = re.compile(r'[\w|\d]+')
            words = ''.join(pattern.findall(m))
            raw_train_data[n][i] = cut(words)
    raw_train_data.to_csv(config.train_seg_path)
