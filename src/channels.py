import tensorflow as tf
from keras.layers import Layer
   

class InsertionDeletionSubstitutionGallager(Layer):
    def __init__(self, i=0.0, d=0.1, s=0.0, pad_symbol=2, pad_length=64, alphabet=2, **kwargs):
        super(InsertionDeletionSubstitutionGallager, self).__init__(**kwargs)
        self.i = i
        self.d = d
        self.s = s
        self.alphabet = alphabet
        self.pad_symbol = pad_symbol
        self.pad_length = pad_length
        print(f"IDS with ({i}, {d}, {s}), pad_symbol={pad_symbol}, pad_length={pad_length}")

    def call(self, inputs, *args, **kwargs):
        x = inputs
        B, N, _ = x.shape

        noise_distribution_logits = tf.math.log([[self.i, self.d, self.s, 1 - self.i - self.d - self.s]])
        def pad_to_length(v, L, value):
            pad = [[0, tf.maximum(L - tf.shape(v)[0], 0)]]
            padded = tf.pad(v, pad, constant_values=value)
            padded = tf.ensure_shape(padded, [self.pad_length])

            tf.debugging.assert_equal(tf.shape(padded)[0], self.pad_length)
            return padded

        
        def _call_seq(seq):
            mask = tf.reshape(tf.random.categorical(noise_distribution_logits, N), [1, N])
            mask_rep = tf.repeat(mask, 2, axis=1)
            x_rep = tf.repeat(seq, 2, axis=1)
            x_with_i = tf.where(mask_rep == 0, 
                                tf.random.uniform(shape=tf.shape(mask_rep), minval=0, maxval=self.alphabet, dtype=tf.int32), 
                                x_rep)
            x_with_is = tf.where(mask_rep == 2, 
                                 tf.math.floormod(x_with_i + tf.random.uniform(shape=tf.shape(x_with_i), minval=1, maxval=self.alphabet, dtype=tf.int32), self.alphabet), 
                                 x_with_i)
            even_indices_mask = tf.math.floormod(tf.range(2*N)[None,:],2)==0
            gather_ind = tf.logical_or(tf.logical_and(tf.equal(mask_rep, 3), even_indices_mask), 
                                       tf.logical_or(tf.equal(mask_rep, 0), 
                                                     tf.logical_and(tf.equal(mask_rep,2), even_indices_mask)))
            x_gathered = tf.boolean_mask(x_with_is, gather_ind)

            return pad_to_length(x_gathered, self.pad_length, self.pad_symbol)[None, ...]
        

        

        y = tf.concat([_call_seq(x_[...,0]) for x_ in tf.split(x, num_or_size_splits=B, axis=0)], axis=0)[..., None]

        return y
    
    def build(self, input_shape):
        super().build(input_shape)  