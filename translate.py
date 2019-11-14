import tensorflow as tf
import os
import pandas as pd
from config import config
from utilities.GPU_settings import gpu_memory_limit
from gensim.models import Word2Vec
from Model_Seq2Seq import Encoder, Decoder
from embed import load_train, get_embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences


gpus = tf.config.experimental.list_physical_devices('GPU')
gpu_memory_limit()  # When errors come "Fail to find the dnn implementation." put this on

BATCH_SIZE = 8
embedding_dim = 256
units = 200
vocab_size = 20000

_, _, word_index1, word_index2, tokenizer1, tokenizer2 = load_train()

# 使用预训练的词向量
w2v_model = Word2Vec.load(config.w2v_bin_path)
input_weights, output_weights = get_embedding(w2v_model)

encoder = Encoder(vocab_size, embedding_dim, units, BATCH_SIZE, input_weights)
decoder = Decoder(vocab_size, embedding_dim, units, BATCH_SIZE, output_weights)
optimizer = tf.keras.optimizers.Adam()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def beam_deduct(sentence, width=2, max_len=50):
    # Beam Search applied
    # width determine the beam width of one search, which means how many search threads to compare
    # max_len determine the max sequence length of output
    # Note: more time-consuming & space_consuming than greedy search
    beam_width = width
    input_tensor = []
    cut = sentence.split()
    for word in cut:
        if word not in word_index1 or word_index1[word] >= vocab_size:
            continue
        input_tensor.append(word_index1[word])
    # input_tensor shape == [1, 500]
    input_tensor = pad_sequences([input_tensor], padding='post', maxlen=500)
    input_tensor = tf.convert_to_tensor(input_tensor)
    init_hidden = tf.zeros((1, units))
    # for single dedcution, the sentence will only be processed by encoder once
    enc_output, enc_hidden = encoder(input_tensor, init_hidden)
    dec_hidden = enc_hidden
    # here notice the squre brackets, without it, dec_input shape == [1, ], with it, shape == [1,1]
    dec_input = tf.expand_dims([tokenizer2.word_index['<start>']], 0)
    # passing <start> to decoder, and make sure the first token is '<start>'
    _, dec_hidden, attention_weight = decoder(dec_input, dec_hidden, enc_output)
    dec_input = tf.expand_dims([tokenizer2.word_index['<start>']], 0)
    prediction, dec_hidden, attention_weight = decoder(dec_input, dec_hidden, enc_output)
    # sort by probability of each word on vocab, select beam_width number as candidate
    prob_sorted = tf.argsort(tf.nn.softmax(prediction[0]), direction='DESCENDING')
    word_candid = prob_sorted[:beam_width]
    word_prob = [tf.nn.softmax(prediction[0])[x] for x in word_candid]
    result = ['<start> ']*beam_width
    # first word is out of the loop
    for i in range(beam_width):
        result[i] += tokenizer2.index_word[word_candid.numpy()[i]] + ' '
    # for each word_id do the greedy search and sum the probabily divide with step length
    # before <end> to ease loss on longer sequence.
    for j, word_id in enumerate(word_candid):
        word_id = tf.expand_dims([word_id], 0)
        length = 0
        for i in range(max_len-2):
            prediction, dec_hidden, _ = decoder(word_id, dec_hidden, enc_output)
            max_word = tf.argmax(prediction[0]).numpy()
            prob = tf.nn.softmax(prediction[0])[max_word]
            # sum sequential probabilities
            word_prob[j] += prob
            result[j] += tokenizer2.index_word[max_word] + ' '
            if tokenizer2.index_word[max_word] == '<end>':
                break
            word_id = tf.expand_dims([max_word], 0)
            length += 1
        word_prob[j] /= length
    word_prob = tf.argmax(word_prob)
    result = result[word_prob]
    return result


def greedy_deduct(sentence, max_len):
    # greedy search  -- alwasy choose the best logits answer
    # least time & space consuming, but with shortcoming on result.
    input_tensor = []
    result = ''
    cut = sentence.split()
    for word in cut:
        if word not in word_index1 or word_index1[word] >= vocab_size:
            continue
        input_tensor.append(word_index1[word])
    # input_tensor shape == [1, 500]
    input_tensor = pad_sequences([input_tensor], padding='post', maxlen=500)
    input_tensor = tf.convert_to_tensor(input_tensor)
    init_hidden = tf.zeros((1, units))
    # for single dedcution, the sentence will only be processed by encoder once
    enc_output, enc_hidden = encoder(input_tensor, init_hidden)
    dec_hidden = enc_hidden
    # here notice the squre brackets, without it, dec_input shape == [1, ], with it, shape == [1,1]
    dec_input = tf.expand_dims([tokenizer2.word_index['<start>']], 0)
    for i in range(max_len-1):
        prediction, dec_hidden, attention_weight = decoder(dec_input, dec_hidden, enc_output)
        prediction_id = tf.argmax(prediction[0]).numpy()
        # collect words for result
        result += tokenizer2.index_word[prediction_id] + ' '
        if tokenizer2.index_word[prediction_id] == '<end>':
            return result
        # feed the next round dec_input with prediction_id
        dec_input = tf.expand_dims([prediction_id], 0)
    return result


def translate(sentence):
    # prediction = beam_deduct(sentence, width=3, max_len=50)
    prediction = greedy_deduct(sentence, max_len=50)
    print(f"Original Input:{sentence}")
    print(f"Report generated: {prediction}")


def generate_test_csv(input_path, output_path):
    input = pd.read_csv(input_path, encoding='utf-8')
    output = pd.read_csv(output_path, encoding='utf-8')
    data = input['Input']
    report = []
    print(len(data))
    for i, sentence in enumerate(data):
        if i % 200 == 0:
            print("Translation at {} place, process {:.2f}%".format(i, 100*i/len(data)))
        prediction = greedy_deduct(sentence, max_len=50)
        # prediction = beam_deduct(sentence, width=3, max_len=50)
        report.append(prediction)
    report = pd.Series(report, name='Report')
    output.insert(loc=5, column='Report', value=report)
    output.to_csv(config.test_result_path, encoding='utf-8', index=False)


def remove_token(path):
    token = ['<start>', '<end>']
    df = pd.read_csv(path, encoding='utf-8')
    reports = df['Report'].values.tolist()
    removed = []
    for report in reports:
        words = str(report).split()
        words = [word for word in words if word not in token]
        words = ''.join(words)
        removed.append(words)
    removed = pd.Series(removed, name='Report')
    df['Report'] = removed
    df.to_csv('./Data_set/removed_result.csv', encoding='utf-8', index=False)


translate('奥迪 一汽大众 奥迪 <start> 修 一下 钱 换 修 技师 你好 师傅 抛光 处理 一下 50 元 左右 希望 能够 帮到 祝 愉快 <end>')
generate_test_csv(config.testdata_path, config.test_path)
remove_token(config.test_result_path)