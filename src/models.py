import tensorflow as tf
import keras
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Model
from src.sc_ops import SplitEvenOdd, Interleave
from src.utils import MeanTensor

keras.backend.set_floatx('float32')
dtype = keras.backend.floatx()


class NeuralPolarDecoder(Model):
    def __init__(self, embedding_nn, checknode_nn, bitnode_nn, emb2llr_nn, embedding_labels_nn, build_metrics=True):
        super(NeuralPolarDecoder, self).__init__()

        self.embedding_observations_nn = embedding_nn
        self.checknode_nn = checknode_nn
        self.bitnode_nn = bitnode_nn
        self.emb2llr_nn = emb2llr_nn
        self.embedding_labels_nn = embedding_labels_nn
        self.interleave = Interleave(axis=1)
        self.split_even_odd = SplitEvenOdd(axis=1)
        self.eps = 1e-6

        self.loss_fn = SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        if build_metrics:
            self.synthetic_channel_entropy_metric = MeanTensor(name="synthetic_channel_entropy")

    def call(self, inputs, training=False, **kwargs):
        x, y = inputs
        tf.debugging.assert_rank(x, 3)
        tf.debugging.assert_rank(y, 3)

        v = tf.cast(x, tf.int32)
        e = self.embedding_observations_nn(y)
        e = tf.ensure_shape(e, [x.shape[0], x.shape[1], e.shape[-1]])
        loss_array = list()
        norm_array = list()
        pred_array = list()

        V = list([v])
        E = list([e])

        depth = e.shape[1]
        num_of_splits = 1
        while depth > 1:
            V_1 = list([])
            V_2 = list([])
            E_1 = list([])
            E_2 = list([])
            for v, e in zip(V, E):
                # compute bits amd embeddings in next layer
                v_odd, v_even = self.split_even_odd.call(v)
                V_1.append(v_odd)
                V_2.append(v_even)
                e_odd, e_even = self.split_even_odd.call(e)
                E_1.append(e_odd)
                E_2.append(e_even)

            V_odd = tf.concat(V_1, axis=1)
            V_even = tf.concat(V_2, axis=1)
            v_xor = tf.math.floormod(V_odd + V_even, 2)
            V_xor = tf.split(v_xor, num_or_size_splits=2 ** (num_of_splits - 1), axis=1)
            V_identity = tf.split(V_even, num_or_size_splits=2 ** (num_of_splits - 1), axis=1)

            v = tf.concat([elem for pair in zip(V_xor, V_identity) for elem in pair], axis=1)
            V_ = tf.split(v, num_or_size_splits=2 ** num_of_splits, axis=1)
            E_odd = tf.concat(E_1, axis=1)
            E_even = tf.concat(E_2, axis=1)
            V_left = self.embedding_labels_nn(tf.squeeze(v_xor, axis=-1))
            e1_left = self.checknode_nn.call(tf.concat((E_odd, E_even), axis=-1))
            e1_right = self.bitnode_nn.call(tf.concat((E_odd, E_even, V_left), axis=-1))
            E1_left = tf.split(e1_left, num_or_size_splits=2 ** (num_of_splits - 1), axis=1)
            E1_right = tf.split(e1_right, num_or_size_splits=2 ** (num_of_splits - 1), axis=1)
            e_lr = tf.concat([elem for pair in zip(E1_left, E1_right) for elem in pair], axis=1)

            E_ = tf.split(e_lr, num_or_size_splits=2 ** num_of_splits, axis=1)


            pred_ = self.emb2llr_nn.__call__(tf.concat(E_, axis=1))
            loss_ = self.loss_fn(tf.concat(V_, axis=1), pred_)
            norms = tf.norm(e, ord='euclidean', axis=-1)
            loss_array.append(loss_)
            pred_array.append(pred_)
            norm_array.append(norms)
            V = V_
            E = E_

            depth //= 2
            num_of_splits += 1
        return tf.stack(loss_array, axis=-2), tf.stack(pred_array, axis=-2), tf.stack(pred_array, axis=-2)

    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            loss_array, pred_array, norm_array = self(inputs, training=True)

            loss = tf.reduce_mean(loss_array)

        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        ce = loss_array[:,-1,:] / tf.math.log(2.0)
        self.synthetic_channel_entropy_metric.update_state(tf.reduce_mean(ce, axis=0))
        res = {'loss': loss / tf.math.log(2.0),
               'mi': 1.0 - tf.reduce_mean(self.synthetic_channel_entropy_metric.result()),
               'grad_norm': tf.linalg.global_norm(gradients)}
        return res

    @tf.function
    def test_step(self, inputs):
        loss_array, pred_array, norm_array = self(inputs, training=False)
        ce = loss_array[:,-1,:] / tf.math.log(2.0)
        self.synthetic_channel_entropy_metric.update_state(ce)
        # Return a dict mapping metric names to current value
        res = {'ce': tf.reduce_mean(self.synthetic_channel_entropy_metric.result()),
               }
        return res

    def build(self, input_shape):
        super().build(input_shape)


class NeuralPolarDecoderHondaYamamoto(Model):
    def __init__(self, npd_const, npd_channel, build_metrics=True):
        super(NeuralPolarDecoderHondaYamamoto, self).__init__()
        self.npd_const = npd_const
        self.npd_channel = npd_channel

        if build_metrics:
            self.synthetic_channel_entropy_metric_x = MeanTensor(name="synthetic_channel_entropy_x")
            self.synthetic_channel_entropy_metric_y = MeanTensor(name="synthetic_channel_entropy_y")

    def call(self, inputs):
        x, y = inputs
        inputs_tilde = x, 2 * tf.ones_like(y)
        loss_array_x, _, _ = self.npd_const(inputs_tilde, training=False)
        loss_array_y, _, _ = self.npd_channel(inputs, training=False)
        return loss_array_x, loss_array_y

    @tf.function
    def train_step(self, inputs):
        x, y = inputs
        inputs_tilde = x, 2*tf.ones_like(y)
        with tf.GradientTape() as tape:
            loss_array_x, _, _ = self.npd_const(inputs_tilde, training=True)
            loss_array_y, _, _ = self.npd_channel(inputs, training=True)

            loss = tf.reduce_mean(loss_array_x + loss_array_y)


        gradients = tape.gradient(loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        ce_x = loss_array_x[:,-1,:] / tf.math.log(2.0)
        ce_y = loss_array_y[:,-1,:] / tf.math.log(2.0)

        self.synthetic_channel_entropy_metric_x.update_state(ce_x)
        self.synthetic_channel_entropy_metric_y.update_state(ce_y)
        res = {'loss': loss / tf.math.log(2.0),
               'ce_x': tf.reduce_mean(self.synthetic_channel_entropy_metric_x.result()),
               'ce_y': tf.reduce_mean(self.synthetic_channel_entropy_metric_y.result()),
               'mi': tf.reduce_mean(self.synthetic_channel_entropy_metric_x.result() - self.synthetic_channel_entropy_metric_y.result()),
               }
        return res

    @tf.function
    def test_step(self, inputs):
        x, y = inputs
        inputs_tilde = x, 2 * tf.ones_like(y)
        loss_array_x, _, _ = self.npd_const(inputs_tilde, training=False)
        loss_array_y, _, _ = self.npd_channel(inputs, training=False)

        ce_x = loss_array_x[:,-1,:] / tf.math.log(2.0)
        ce_y = loss_array_y[:,-1,:] / tf.math.log(2.0)

        self.synthetic_channel_entropy_metric_x.update_state(ce_x)
        self.synthetic_channel_entropy_metric_y.update_state(ce_y)
        res = {'ce_x': self.synthetic_channel_entropy_metric_x.result(),
               'ce_y': self.synthetic_channel_entropy_metric_y.result(),
               'mi': self.synthetic_channel_entropy_metric_x.result() - self.synthetic_channel_entropy_metric_y.result(),
               }
        return res

    def build(self, input_shape):
        super().build(input_shape)


class NeuralPolarDecoderOptimize(Model):
    def __init__(self, npd_const, npd_channel, input_distribution, channel, build_metrics=True):
        super(NeuralPolarDecoderOptimize, self).__init__()
        self.npd_const = npd_const
        self.npd_channel = npd_channel
        self.input_distribution = input_distribution
        self.channel = channel

        if build_metrics:
            self.synthetic_channel_entropy_metric_x = MeanTensor(name="synthetic_channel_entropy_x")
            self.synthetic_channel_entropy_metric_y = MeanTensor(name="synthetic_channel_entropy_y")

        self.opt_est = None
        self.opt_improve = None

    def call(self, inputs):
        batch, N = inputs.shape
        x, _ = self.input_distribution.generate(
            length=N,
            batch_size=batch
        )

        loss_array_x, _, _ = self.npd_const((x, 2 * tf.ones_like(x)), training=True)
        loss_array_y, _, _ = self.npd_channel((x, x), training=True)

    def compile(self, opt_est, opt_impr, **kwargs):
        super().compile(**kwargs)
        self.opt_est = opt_est
        self.opt_improve = opt_impr

    @tf.function
    def train_step(self, inputs):
        batch, N = inputs.shape
        x, _ = self.input_distribution.generate(
            length=N,
            batch_size=batch
        )
        x = tf.stop_gradient(x)
        y = self.channel(x)
        with tf.GradientTape() as tape:
            loss_array_x, _, _ = self.npd_const((x, 2*tf.ones_like(y)), training=True)
            loss_array_y, _, _ = self.npd_channel((x, y), training=True)
            loss_est = tf.reduce_mean(loss_array_x + loss_array_y)

        gradients = tape.gradient(loss_est, self.npd_const.trainable_weights + self.npd_channel.trainable_weights)

        ce_x = loss_array_x[:, -1, :] / tf.math.log(2.0)
        ce_y = loss_array_y[:, -1, :] / tf.math.log(2.0)

        self.synthetic_channel_entropy_metric_x.update_state(ce_x)
        self.synthetic_channel_entropy_metric_y.update_state(ce_y)

        # Update weights
        self.opt_est.apply_gradients(zip(gradients, self.npd_const.trainable_weights + self.npd_channel.trainable_weights))

        with tf.GradientTape() as tape:
            x, llr_x1 = self.input_distribution.generate(
                length=N,
                batch_size=batch
            )
            y = self.channel(x)
            x, y = tf.stop_gradient(x), tf.stop_gradient(y)
            loss_array_x, _, _ = self.npd_const((x, 2 * tf.ones_like(y)), training=True)
            loss_array_y, _, _ = self.npd_channel((x, y), training=True)

            ce_x = loss_array_x[:, -1, :] / tf.math.log(2.0)
            ce_y = loss_array_y[:, -1, :] / tf.math.log(2.0)

            mi = tf.reduce_mean(ce_x - ce_y, axis=-1)
            mi_mean = tf.reduce_mean(mi)
            # reward = mi - mi_mean
            reward = (mi - mi_mean) / (tf.math.sqrt( tf.reduce_mean( tf.square(mi - mi_mean )) + 1e-10))

            # reward = tf.reduce_mean( tf.square(ce_x - ce_y - mi_mean )) * 0.1  # Regularization term to encourage polarization

            # mi = ce_x - ce_y
            # mi_mean = tf.reduce_mean(mi, axis=0, keepdims=True)
            # reward_arg = (mi - mi_mean) / (tf.math.sqrt( tf.reduce_mean( tf.square(mi - mi_mean ), axis=0, keepdims=True) + 1e-10))
            # reward = tf.reduce_mean(reward_arg, axis=-1)

            llr_x = tf.where(tf.equal(x, 1), llr_x1, -llr_x1)
            log_px = tf.math.log(tf.math.sigmoid(llr_x) + 1e-10)
            log_px_N = tf.reduce_mean(log_px, axis=(1, 2)) 
            # log_px_N = tf.reduce_sum(log_px, axis=(1, 2)) / tf.math.sqrt(tf.cast(N, tf.float32))
            loss_improve = tf.reduce_mean(-tf.stop_gradient(reward) * log_px_N)

        gradients = tape.gradient(loss_improve, self.input_distribution.trainable_weights)
        # Update weights
        self.opt_improve.apply_gradients(
            zip(gradients, self.input_distribution.trainable_weights))

        res = {
            'mi': tf.reduce_mean(self.synthetic_channel_entropy_metric_x.result() - self.synthetic_channel_entropy_metric_y.result()),
            'ce_x': tf.reduce_mean(self.synthetic_channel_entropy_metric_x.result()),
            'ce_y': tf.reduce_mean(self.synthetic_channel_entropy_metric_y.result()),
            'loss_est': loss_est / tf.math.log(2.0),
            'loss_improve': loss_improve,
        }
        return res

    @tf.function
    def test_step(self, inputs):
        batch, N = inputs.shape
        x, _ = self.input_distribution.generate(
            length=N,
            batch_size=batch
        )
        y = self.channel(x)
        inputs_tilde = x, 2 * tf.ones_like(y)
        loss_array_x, _, _ = self.npd_const(inputs_tilde, training=False)
        loss_array_y, _, _ = self.npd_channel((x, y), training=False)

        ce_x = loss_array_x[:,-1,:] / tf.math.log(2.0)
        ce_y = loss_array_y[:,-1,:] / tf.math.log(2.0)

        self.synthetic_channel_entropy_metric_x.update_state(ce_x)
        self.synthetic_channel_entropy_metric_y.update_state(ce_y)
        res = {'ce_x': tf.reduce_mean(self.synthetic_channel_entropy_metric_x.result()),
               'ce_y': tf.reduce_mean(self.synthetic_channel_entropy_metric_y.result()),
               'mi': tf.reduce_mean(self.synthetic_channel_entropy_metric_x.result() - self.synthetic_channel_entropy_metric_y.result()),
               }
        return res

    def build(self, input_shape):
        super().build(input_shape)
