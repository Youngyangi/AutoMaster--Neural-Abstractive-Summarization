import os
import tensorflow as tf
import time
from config import config
from gensim.models import Word2Vec
from Model_Seq2Seq import Encoder, Decoder, loss_function
from embed import get_embedding, load_train

# '1'显示全部， ’2‘显示error和warning， ’3‘只显示error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

BATCH_SIZE = 8
embedding_dim = 256
units = 200
vocab_size = 20000
EPOCH = 10
using_GPU = True


# 使用预训练的词向量
w2v_model = Word2Vec.load(config.w2v_bin_path)
input_weights, output_weights = get_embedding(w2v_model)

input_tensor_train, target_tensor_train, word_index1, word_index2, tokenizer1, tokenizer2 = load_train()

BUFFER_SIZE = len(input_tensor_train)
steps_per_epoch = BUFFER_SIZE//BATCH_SIZE

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

# example_input_batch, example_target_batch = next(iter(dataset))
# print(example_input_batch.shape, example_target_batch.shape)

# 实例化
encoder = Encoder(vocab_size, embedding_dim, units, BATCH_SIZE, input_weights)
decoder = Decoder(vocab_size, embedding_dim, units, BATCH_SIZE, output_weights)
# 定义优化器和损失函数
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
# 定义checkpoint
checkpoint_dir = './training_checkpoints'
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
checkpoint_mng = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=3,
                                            checkpoint_name='20kvocab_256dim_200unit')


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
            if batch % 100 == 0 and batch != 0:
                print('Epoch {} Batch {} at {} Sample, Loss {:.4f}'.format(epoch + 1, batch,
                                                                           batch*BATCH_SIZE, batch_loss.numpy()))
            if batch % 2000 == 0:
                checkpoint_mng.save()
                print('Checkpoint saved')
        if (epoch + 1) % 2 == 0:
            checkpoint_mng.save()
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


try:
    checkpoint.restore(checkpoint_mng.latest_checkpoint)
    print("Restored from latest checkpoint")
    print("Start training...")
    run_op(EPOCH)
except KeyboardInterrupt:
    checkpoint_mng.save()
    print("KeyboardInterrupted, Checkpoint saved")

