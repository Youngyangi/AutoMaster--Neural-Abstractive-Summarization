import keras
import tensorflow as tf


class Encoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, matrix, enc_units, batch_size):
        super().__init__()
        # super(Encoder, self).__init__()
        weights = [matrix]
        self.bc_size = batch_size
        self.enc_units = enc_units
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim, weights=weights, trainable=False)
        self.gru = keras.layers.GRU(self.enc_units,
                                    return_state=True,
                                    return_sequences=True,
                                    recurrent_initializer='glorot_uniform')

    def call(self, sequence, states):
        embed = self.embedding(sequence)
        output, state_h = self.gru(embed, initial_state=states)
        # output = [batch_size, max_length, enc_units]
        # state_h = [batch_size, enc_units]
        return output, state_h

    def init_hidden_state(self):
        return tf.zeros((self.bc_size, self.enc_units))


class BahdanauAttention(keras.layers.Layer):
    def __init__(self, enc_units):
        super().__init__()
        self.W1 = keras.layers.Dense(enc_units)
        self.W2 = keras.layers.Dense(enc_units)
        self.V = keras.layers.Dense(1)

    def call(self, query, value):
        # query: state_h
        # value: output
        hidden_with_time_axis = tf.expand_dims(query, 1)
        # hidden_with_time_axis = [batch_size, 1, enc_units]
        score = self.V(tf.nn.tanh(self.W1(value) + self.W2(hidden_with_time_axis)))
        # self.W2(hidden_with_time_axis): [batch_size, 1, enc_units]
        # self.W1(values): [batch_size, max_len, enc_units]
        # score: [batch_size, max_len, 1]
        # attention_weights对score的1号axis求softmax, 即做出一句话中每个词的权重分布
        attention_weights = tf.nn.softmax(score, axis=1)
        # 对输出的outpu中每个词乘上对应的权重
        context_vec = attention_weights * value
        # 同样每句话中逐词求和，得到context_vector shape == [batch_size, enc_units]
        context_vec = tf.reduce_sum(context_vec, axis=1)
        return context_vec, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, matrix, dec_units, batch_size):
        super().__init__()
        # super(Decoder, self).__init__()
        weights = [matrix]
        self.bc_size = batch_size
        self.dec_units = dec_units
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim, weights=weights, trainable=False)
        self.gru = keras.layers.GRU(self.dec_units,
                                    return_state=True,
                                    return_sequences=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(dec_units)

    def call(self, x, hidden, enc_output):
        embed = self.embedding(x)
        context_vec, attention_weight = self.attention(hidden, enc_output)
        # expand_dims后的context_vec shape == [batch_size, 1, dec_units]
        # embed shape == [batch_size, 1, embedding_dim]
        # dec_input shape == [batch_size, 1, dec_units + embedding_dim]
        dec_input = tf.concat([tf.expand_dims(context_vec, axis=1), embed], axis=-1)
        # passing to GRU
        output, state = self.gru(dec_input)
        # output shape == [batch_size , 1, dec_units]
        # after reshape, output shape == [batchsize, dec_dec_units]
        output = tf.reshape(output, (-1, output.shape[2]))
        # output = tf.squeeze(output, axis=1)
        logits = self.fx(output)
        return logits, state, attention_weight


