import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Layer, Dense, Dropout, Embedding, Conv1D, MultiHeadAttention, Lambda
from keras.initializers import GlorotNormal, GlorotUniform


class NodeNN(Model):
    def __init__(self, hidden_dim, embedding_dim, layers, activation='elu',
                 use_bias=True, dropout=0., name='node_nn', **kwargs):
        super(NodeNN, self).__init__(name=name, **kwargs)

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = layers
        self.activation = activation
        self.use_bias = use_bias
        self.dropout_rate = dropout

        self._layers = [Dense(hidden_dim, activation=activation, use_bias=use_bias,
                              kernel_initializer=GlorotNormal(), name=f"{name}-layer{i}")
                        for i in range(layers)] + \
                       [Dense(embedding_dim, activation=None, use_bias=use_bias,
                              kernel_initializer=GlorotNormal(), name=f"{name}-layer{layers}")]

        self.dropout = Dropout(dropout)
        self.layer_norm = RMSNorm()



    def call(self, inputs, training=None, *args):
        e = inputs
        # e = self.layer_norm(e)
        for layer in self._layers:
            e = layer(e, training=training)
            e = self.dropout(e, training=training)
        return self.layer_norm(e)

    def get_config(self):
        config = super(NodeNN, self).get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
            "layers": self.num_layers,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "dropout": self.dropout_rate,
            "name": self.name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Embedding2Prob(Model):
    def __init__(self, q=2, activation='softmax', use_bias=True, name='emb2llr_nnops'):
        super(Embedding2Prob, self).__init__(name=name)
        assert activation=='softmax' or activation is None, \
            f"invalid activation type for embedding to prob layer: {activation}"
        self.layer = Dense(q, use_bias=use_bias, activation=activation,
                           kernel_initializer=GlorotNormal(),)

    def call(self, inputs, training=None, **kwargs):
        e = inputs
        e = self.layer.__call__(e, training=training)
        return e

    def get_config(self):
        config = super(Embedding2Prob, self).get_config()
        config.update({
            "q": self.layer.units,
            "activation": self.layer.activation.__name__,
            "use_bias": self.layer.use_bias,
        })
        return config


class ConstEmbedding(Layer):
    def __init__(self, embedding_dim, **kwargs):
        super(ConstEmbedding, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.emb = self.add_weight(
            shape=(self.embedding_dim,),
            initializer=GlorotNormal(),
            trainable=True,
            name="const_embedding"
        )

    def call(self, inputs):
        B, N = tf.shape(inputs)[0], tf.shape(inputs)[1]
        emb_tiled = tf.reshape(self.emb, (1, 1, self.embedding_dim))  # shape (1, 1, d)
        return tf.tile(emb_tiled, (B, N, 1))  # shape (B, N, d)

    def get_config(self):
        config = super().get_config()
        config.update({"embedding_dim": self.embedding_dim})
        return config


class CNNEmbedding(Model):
    def __init__(self, block_length, embedding_size, hidden_size, layers, output_alphabet=3,  kernel_size=16, channel_embedding_size=4,
                 activation='elu', strides=1, padding='same'):
        super(CNNEmbedding, self).__init__()

        self.N = block_length
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides
        self.padding = padding
        self.layers_count = layers
        self.padding = padding

        self.embedding_layer = Embedding(input_dim=output_alphabet, output_dim=channel_embedding_size)
        self.pos_embedding = PositionalEmbedding(block_length, channel_embedding_size)

        self.model_layers = Sequential([Conv1D(filters=hidden_size,
                                         kernel_size=kernel_size,
                                         strides=strides,
                                         activation=activation,
                                         padding=padding,
                                         kernel_initializer=GlorotNormal()) for _ in range(self.layers_count)] +
                                 [Conv1D(filters=embedding_size,
                                         kernel_size=kernel_size,
                                         strides=strides,
                                         activation=None,
                                         padding=padding,
                                         kernel_initializer=GlorotNormal())]
                                 )

        self.layer_norm = RMSNorm()


    def call(self, inputs, *args, **kwargs):
        y_split = tf.split(inputs, num_or_size_splits=inputs.shape[-1], axis=-1)

        es = [self.embedding_layer(tf.squeeze(y, axis=-1)) for y in y_split]
        es_pos = [self.pos_embedding(e) for e in es] 
        es_trellis = [self.model_layers(e) for e in es_pos]
        es_stacked = tf.stack(es_trellis, axis=-1)
        output = tf.reduce_sum(es_stacked, axis=-1) / tf.sqrt(tf.cast(inputs.shape[-1], tf.float32))
        return self.layer_norm(output)

    def get_config(self):
        config = super().get_config()
        config.update({"block_length": self.block_length,
                       "embedding_size": self.embedding_dim,
                       "hidden_size": self.hidden_size,
                       "layers": self.layers_count,
                       "activation": self.activation,
                       "output_alphabet": self.output_alphabet,
                       "strides": 1,
                       "padding": 'same'})
        return config


class AttentionEmbedding(Model):
    def __init__(self, block_length, embedding_size, channel_embedding_size, hidden_size,
                 output_alphabet=3, layers=2, num_heads=16, activation='elu'):
        super(AttentionEmbedding, self).__init__()
        self.block_length = block_length
        self.positional_embedding_max_length = 2 * block_length
        self.embedding_size = embedding_size
        self.channel_embedding_size = channel_embedding_size
        self.layers_count = layers
        self.num_heads = num_heads
        self.activation = activation
        self.output_alphabet = output_alphabet

    
        self.channel_embedding_model = Sequential([Lambda(lambda x: tf.squeeze(x, axis=-1)),
            Embedding(input_dim=output_alphabet, output_dim=channel_embedding_size),
                                                   PositionalEmbedding(self.positional_embedding_max_length, self.channel_embedding_size)])


        self.attention_blocks = [AttentionEmbeddingLayer(num_heads=self.num_heads, 
                                                         embed_dim=self.embedding_size, 
                                                         hidden_dim=hidden_size,
                                                         activation=activation)
                                 for _ in range(self.layers_count)]

    def call(self, inputs, *args, **kwargs):
        inputs = self.channel_embedding_model(inputs)
        V = K = inputs
        Q = self.channel_embedding_model((self.output_alphabet-1) * tf.ones(shape=(inputs.shape[0], self.block_length,)))

        output = Q
        # Generate constant embedding with positional encoding

        # Apply multi-head cross-attention
        for layer in self.attention_blocks:
            output = layer((output, K, V))

        return output

    def build(self, input_shape):
        super().build(input_shape)    

    def get_config(self):
        config = super(AttentionEmbedding, self).get_config()
        config.update({
            "block_length": self.block_length,
            "embedding_size": self.embedding_size,
            "channel_embedding_size": self.channel_embedding_size,
            "layers": self.layers_count,
            "num_heads": self.num_heads,
            "activation": self.activation,
            "output_alphabet": self.output_alphabet,
        })
        return config


class AttentionEmbeddingLayer(Layer):
    def __init__(self, num_heads, embed_dim, hidden_dim, activation='relu'):
        super(AttentionEmbeddingLayer, self).__init__()
        self.embedding_size = embed_dim
        self.self_attention = AttentionBlock(num_heads, embed_dim)
        self.cross_attention = AttentionBlock(num_heads, embed_dim)
        self.dense = FCBlock(embed_dim, hidden_dim, activation=activation)

    def call(self, inputs, *args, **kwargs):
        Q, K, V = inputs
        output = self.self_attention((Q,Q,Q))
        output = self.cross_attention((output, K, V))
        output = self.dense(output)
        return output

    def build(self, input_shape):
        super().build(input_shape) 

    def get_config(self):
        config = super(AttentionEmbeddingLayer, self).get_config()
        config.update({
            "num_heads": self.self_attention.mha.num_heads,
            "embed_dim": self.embedding_size,
            "hidden_dim": self.dense.dense.layers[0].units,
            "activation": self.dense.dense.layers[0].activation.__name__,
        })
        return config


class AttentionBlock(Layer):
    def __init__(self, num_heads, embed_dim):
        super(AttentionBlock, self).__init__()
        self.embedding_size = embed_dim
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm_mha = RMSNorm()

    def call(self, inputs, *args, **kwargs):
        Q, K, V = inputs
        # Apply multi-head cross-attention
        attention_output = self.mha(query=Q, key=K, value=V)
        attention_output = self.norm_mha(Q + attention_output)
        return attention_output

    def build(self, input_shape):
        super().build(input_shape)

    def get_config(self):
        config = super(AttentionBlock, self).get_config()
        config.update({
            "num_heads": self.mha.num_heads,
            "embed_dim": self.embedding_size,
        })
        return config
    

class FCBlock(Layer):
    def __init__(self, embed_dim, hidden_dim, activation='relu'):
        super(FCBlock, self).__init__()
        self.embedding_size = embed_dim
        self.dense = Sequential([Dense(hidden_dim, activation=activation, use_bias=True),
                                 Dense(embed_dim, activation=None, use_bias=True)])
        self.norm_dense = RMSNorm()

    def call(self, inputs, *args, **kwargs):
        # Pass through dense layer
        output = self.dense(inputs)
        output = self.norm_dense(inputs + output)
        return output

    def build(self, input_shape):
        super().build(input_shape)

    def get_config(self):
        config = super(FCBlock, self).get_config()
        config.update({
            "embed_dim": self.embedding_size,
            "hidden_dim": self.dense.layers[0].units,
            "activation": self.dense.layers[0].activation.__name__,
        })
        return config


class PositionalEmbedding(Layer):
    def __init__(self, max_seq_len, d_model):
        super(PositionalEmbedding, self).__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.positional_encoding = self.add_weight(name="positional_encoding", shape=(1, max_seq_len, d_model),
                                                   initializer=GlorotNormal(), trainable=True)

    def call(self, inputs, *args, **kwargs):
        seq_len = tf.shape(inputs)[1]  # Get the actual sequence length of the input
        # Add the positional encoding (truncate to the current sequence length)
        return inputs + self.positional_encoding[:, :seq_len, :]

    def build(self, input_shape):
        super().build(input_shape)
    
    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model,
        })
        return config


class RMSNorm(Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super(RMSNorm, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.gamma = None

    def build(self, input_shape):
        # Learnable scale parameter γ (gamma), one per feature
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),  # Scale parameter per feature
            initializer="ones",
            trainable=True,
            name="gamma"
        )
        super(RMSNorm, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # Compute RMS(x) along the last dimension
        rms = tf.sqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + self.epsilon)
        return (inputs / rms) * self.gamma  # Normalize and scale

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


class DyT(Layer):
    def __init__(self, alpha_0=0.5, **kwargs):
        super(DyT, self).__init__(**kwargs)
        self.alpha_0 = alpha_0
        self.alpha = None
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        self.alpha = self.add_weight(
            shape=(input_shape[-1],),  
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
            name="alpha"
        )
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),  
            initializer="ones",
            trainable=True,
            name="gamma"
        )
        self.beta = self.add_weight(
            shape=(input_shape[-1],),  
            initializer="ones",
            trainable=True,
            name="beta"
        )
        super(RMSNorm, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # Compute RMS(x) along the last dimension
        x = tf.math.tanh(self.alpha * inputs)
        
        return self.gamma * x + self.beta

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config
