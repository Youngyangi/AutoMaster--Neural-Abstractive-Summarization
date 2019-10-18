from config import config
import pandas as pd


def get_length_prob(data, threshold):
    length = len(data)
    counter = 0
    for i in data:
        if len(i) >= threshold:
            counter += 1
    return 1-counter/length


def get_data():
    df1 = pd.read_csv(config.traindata_path, encoding='utf-8')
    df2 = pd.read_csv(config.testdata_path, encoding='utf-8')
    data = []
    for i in [df1, df2]:
        for m in i['Input'].values.tolist():
            data.append(m.split())
    return data


data = get_data()
# 输出长度为500可以覆盖99%的输入数据，因此可以确定最大输入长度为500
print('100 length covers', get_length_prob(data, 100))
print('200 length covers', get_length_prob(data, 200))
print('300 length covers', get_length_prob(data, 300))
print('400 length covers', get_length_prob(data, 400))
print('500 length covers', get_length_prob(data, 500))

