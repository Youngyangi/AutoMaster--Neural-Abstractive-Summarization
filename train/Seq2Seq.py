import keras

class Encoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, encoding_units, batch_size, ):
        super().__init__
        self.bc_size = batch_size
        self.enc_units = encoding_units
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)  # word2vec dictionaru
        self.gru = keras.layers.GRU(self.enc_units,
                                    return_state=True,
                                    return_sequences=True,
                                    recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return keras.initializers.zeros((self.bc_size, self.enc_units))


if __name__ == '__main__':
    encoder = Encoder(300, 256, 1024, 64)
    sample_hidden = encoder.initialize_hidden_state()
