from config import config
import jieba
import pandas as pd
import re


def cut(sentence):
    # jieba切词，默认精确模式，全模式cut参数：cut_all=True
    # 加入停用词
    stoplist = ['语音', '|', '图片']
    words = jieba.cut(sentence)
    words = [w for w in words if w not in stoplist]
    return ' '.join(words)


def stopword(path):
    word_list = []
    f = open(path, 'r', encoding='utf-8')
    word_list.extend(f.read().split('\n'))
    word_list = [w.strip() for w in word_list]
    assert '|' in word_list
    f.close()
    return word_list


def build_corpus_text(path):
    traindata = pd.read_csv(config.traindata_path, encoding='utf8')
    testdata = pd.read_csv(config.testdata_path, encoding='utf8')
    with open(path, 'w', encoding='utf-8') as f:
        for m, n in zip(traindata['Input'], testdata['Input']):
            f.write(m + '\n')
            f.write(n + '\n')
        for i in traindata['Report']:
            f.write(i + '\n')
    return


def build_csv(train_path, test_path):
    raw_train_data = pd.read_csv(train_path)
    raw_test_data = pd.read_csv(test_path)
    # 去掉任意带有空值的行，并重新index
    raw_train_data.dropna(axis=0, how='any', inplace=True)
    raw_test_data.dropna(axis=0, how='any', inplace=True)
    raw_train_data.reset_index(inplace=True, drop=True)
    raw_test_data.reset_index(inplace=True, drop=True)
    # 对训练用文本切词后重新放回原位
    for n in ['Model', 'Brand', 'Question', 'Dialogue', 'Report']:
        for i, m in enumerate(raw_train_data[n].values):
            pattern = re.compile(r'[\w|\d]+')
            words = pattern.findall(m)
            words = [w.strip() for w in words]
            words = ''.join(words)
            raw_train_data[n][i] = cut(words)
    for n in ['Model', 'Brand', 'Question', 'Dialogue']:
        for i, m in enumerate(raw_test_data[n].values):
            pattern = re.compile(r'[\w|\d]+')
            words = pattern.findall(m)
            words = [w.strip() for w in words]
            words = ''.join(words)
            raw_test_data[n][i] = cut(words)
    # 进行切词后，可能会出现短句均为停用词情况，再次去掉空值
    raw_train_data.dropna(axis=0, how='any', inplace=True)
    raw_test_data.dropna(axis=0, how='any', inplace=True)
    # 将训练用的文本拼接在一起
    # 加上起止符号<start>, <end>
    raw_train_data['Input'] = raw_train_data['Model'] + ' ' + raw_train_data['Brand'] + ' ' + \
                              '<start> ' + raw_train_data['Question'] + ' ' + raw_train_data['Dialogue'] + ' <end>'
    raw_test_data['Input'] = raw_test_data['Model'] + ' ' + raw_test_data['Brand'] + ' ' + \
                             '<start> ' + raw_test_data['Question'] + ' ' + raw_test_data['Dialogue'] + ' <end>'
    raw_train_data['Report'] = '<start> ' + raw_train_data['Report'] + ' <end>'
    # 去掉多余的列
    raw_train_data.drop(['Model', 'Brand', 'Question', 'Dialogue'], axis=1, inplace=True)
    raw_test_data.drop(['Model', 'Brand', 'Question', 'Dialogue'], axis=1, inplace=True)
    # 保存到csv中
    raw_train_data.to_csv(config.traindata_path, index=False)
    raw_test_data.to_csv(config.testdata_path, index=False)
    return


# build_csv(config.train_path, config.test_path)
build_corpus_text(config.corpus_path)
