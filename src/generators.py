import tensorflow as tf


def info_bits_generator(batch, info_bits_num):
    while True:
        info_bits = tf.random.uniform(shape=(batch, info_bits_num), minval=0, maxval=2, dtype=tf.int32)
        yield info_bits


