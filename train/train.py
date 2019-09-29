import pandas as pd
import config.config


def build_trainset(path1, path2):
    df = pd.read_csv(path1, encoding='utf-8')
    text_series, response_series = [], []
    for i in range(df.shape[0]):
        text = '<start> ' + str(df.loc[i]['Question']) + str(df.loc[i]['Dialogue']) + ' <end>'
        response = '<start> ' + str(df.loc[i]['Report']) + ' <end>'
        text_series.append(text), response_series.append(response)
    trainset = pd.DataFrame({'Text': text_series, 'Response': response_series})
    trainset.to_csv(path2)
    return


if __name__ == '__main__':
    build_trainset(config.config.train_seg_path, config.config.trainset_path)
    df = pd.read_csv(config.config.trainset_path)
    print(df.loc[0]['Text'])

    print(df.head())

