import os
import tensorflow as tf
import time
import jieba
from config import config
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from Model_Seq2Seq import Encoder, Decoder, BahdanauAttention, loss_function
from embed import get_embedding, load_train


BATCH_SIZE = 64
embedding_dim = 100
units = 10
vocab_size = 3000
EPOCH = 10

# 使用预训练的词向量
# w2v_model = Word2Vec.load(config.w2v_bin_path)
# input_weights, output_weights = get_embedding(w2v_model)

input_tensor_train, target_tensor_train, word_index1, word_index2, tokenizer1, tokenizer2 = load_train(num_samples=128)

BUFFER_SIZE = len(input_tensor_train)
steps_per_epoch = BUFFER_SIZE//BATCH_SIZE

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
print(example_input_batch.shape, example_target_batch.shape)
# 实例化
encoder = Encoder(vocab_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_size, embedding_dim, units, BATCH_SIZE)
# 定义优化器和损失函数
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
# 定义checkpoint
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([word_index2['<start>']]*BATCH_SIZE, 1)
        for t in range(targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)
    batch_loss = (loss/int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


def run_op(epochs):
    for epoch in range(epochs):
        start = time.time()
        enc_hidden = encoder.init_hidden_state()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def deduct(sentence):
    input_tensor = []
    result = ''
    cut = sentence.split()
    for word in cut:
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
    for i in range(50):
        prediction, dec_hidden, attention_weight = decoder(dec_input, dec_hidden, enc_output)
        # greedy search here -- alwasy choose the best logits answer
        prediction_id = tf.argmax(prediction[0]).numpy()
        # collect words for result
        result += tokenizer2.index_word[prediction_id] + ' '
        if tokenizer2.index_word[prediction_id] == '<end>':
            return result, sentence
        # feed the next round dec_input with prediction_id
        dec_input = tf.expand_dims([prediction_id], 0)
    return result


def translate(sentence):
    prediction = deduct(sentence)
    print(f"Original Input:{sentence}")
    print(f"Report generated: {prediction}")


# run_op(2)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
translate('奥迪 一汽大众 奥迪 <start> 修 一下 钱 换 修 技师 你好 师傅 抛光 处理 一下 50 元 左右 希望 能够 帮到 祝 愉快 <end>')
