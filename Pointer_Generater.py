import tensorflow as tf


class Pointer(tf.keras.layers.Layer):
    def __init__(self):
        super(Pointer, self).__init__()
        self.w_h = tf.keras.layers.Dense(1, use_bias=False)
        self.w_s = tf.keras.layers.Dense(1, use_bias=False)
        self.w_x = tf.keras.layers.Dense(1, use_bias=False)
        self.w_t = tf.keras.layers.Dense(1, use_bias=True)

    def call(self, dec_input, context_vec, hidden):
        logit = self.w_t(self.w_x(dec_input) + self.w_s(hidden) + self.w_h(context_vec))
        p_gen = tf.nn.sigmoid(logit)
        return p_gen
