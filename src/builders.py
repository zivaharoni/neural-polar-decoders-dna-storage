import tensorflow as tf
from src.models import NeuralPolarDecoder, NeuralPolarDecoderHondaYamamoto, NeuralPolarDecoderOptimize
from keras.models import Sequential
from keras.layers import Dense, Embedding
from src.layers import NodeNN, Embedding2Prob, ConstEmbedding, CNNEmbedding, AttentionEmbedding
from src.input_distribution import BinaryRNN

def build_neural_polar_decoder(config):
    checknode_nn = NodeNN(hidden_dim=config["hidden_dim"],
                          embedding_dim=config['embedding_dim'],
                          layers=config["layers"],
                          activation=config["activation"],
                          use_bias=config["use_bias"],
                          dropout=config["dropout"])
    bitnode_nn = NodeNN(hidden_dim=config["hidden_dim"],
                        embedding_dim=config['embedding_dim'],
                        layers=config["layers"],
                        activation=config["activation"],
                        use_bias=config["use_bias"],
                        dropout=config["dropout"])
    emb2llr_nn = Embedding2Prob()
    embedding_labels_nn = Embedding(input_dim=2,
                                    output_dim=config["embedding_dim"],
                                    trainable=True,
                                    name="symbol_embedding")
    return checknode_nn, bitnode_nn, emb2llr_nn, embedding_labels_nn

def build_cnn_embedding(block_length, embedding_size, hidden_size, output_alphabet, config):
    embedding = CNNEmbedding(block_length=block_length,
                                embedding_size=embedding_size,
                                hidden_size=hidden_size,
                                channel_embedding_size=embedding_size,
                                output_alphabet=output_alphabet,
                                activation=config['cnn_activation'],
                                layers=config['cnn_layers'],
                                kernel_size=max(block_length//4, 4),
                                strides=config['strides'],
                                padding=config['padding'])
    return embedding

def build_attention_embedding(block_length, embedding_size, hidden_size, output_alphabet, config):
    embedding = AttentionEmbedding(block_length=block_length,
                                         embedding_size=embedding_size,
                                         channel_embedding_size=embedding_size,
                                         hidden_size=hidden_size,
                                         activation=config['attention_activation'],
                                         layers=config['attention_layers'],
                                         num_heads=config['heads_num'],
                                         output_alphabet=output_alphabet)
    return embedding

def build_neural_polar_decoder_iid(embedding_config, npd_config, input_shape, load_path=None):
    N = input_shape[0][1]
    if embedding_config["name"] == "cnn":
        embedding_nn_channel = build_cnn_embedding(block_length=N,
                                                   embedding_size=npd_config["embedding_dim"],
                                                   hidden_size=npd_config["hidden_dim"],
                                                   output_alphabet=3,
                                                   config=embedding_config)
    elif embedding_config["name"] == "attention":
        embedding_nn_channel = build_attention_embedding(block_length=N,
                                                   embedding_size=npd_config["embedding_dim"],
                                                   hidden_size=npd_config["hidden_dim"],
                                                   output_alphabet=3,
                                                   config=embedding_config)

    checknode_nn, bitnode_nn, emb2llr_nn, embedding_labels_nn = build_neural_polar_decoder(npd_config)
    model = NeuralPolarDecoder(
                embedding_nn=embedding_nn_channel,
                checknode_nn=checknode_nn,
                bitnode_nn=bitnode_nn,
                emb2llr_nn=emb2llr_nn,
                embedding_labels_nn=embedding_labels_nn,
    )
    model.build(input_shape)

    if load_path is not None:
        try:
            model([tf.zeros(shape, dtype=tf.int32) for shape in input_shape])
            model.load_weights(load_path, skip_mismatch=True)
            print(f"Loaded weights from {load_path}")
        except Exception as e:
                print(e)
                print(f"Model path {load_path} does not exist. Skipping loading weights.")
    return model

def build_neural_polar_decoder_optimize(embedding_config, npd_config, input_shape, channel, load_path=None):
    embedding_nn_const = Sequential([ConstEmbedding(npd_config["embedding_dim"])],
                              name="embedding_const_nn")

    N = input_shape[0][1]
    if embedding_config["name"] == "cnn":
        embedding_nn_channel = build_cnn_embedding(block_length=N,
                                                   embedding_size=npd_config["embedding_dim"],
                                                   hidden_size=npd_config["hidden_dim"],
                                                   output_alphabet=3,
                                                   config=embedding_config)
    elif embedding_config["name"] == "attention":
        embedding_nn_channel = build_attention_embedding(block_length=N,
                                                   embedding_size=npd_config["embedding_dim"],
                                                   hidden_size=npd_config["hidden_dim"],
                                                   output_alphabet=3,
                                                   config=embedding_config)

    checknode_nn, bitnode_nn, emb2llr_nn, embedding_labels_nn = build_neural_polar_decoder(npd_config)


    npd_const = NeuralPolarDecoder(
        embedding_nn=embedding_nn_const,
        checknode_nn=checknode_nn,
        bitnode_nn=bitnode_nn,
        emb2llr_nn=emb2llr_nn,
        embedding_labels_nn=embedding_labels_nn,
                build_metrics=False)
    npd_channel = NeuralPolarDecoder(
        embedding_nn=embedding_nn_channel,
        checknode_nn=checknode_nn,
        bitnode_nn=bitnode_nn,
        emb2llr_nn=emb2llr_nn,
        embedding_labels_nn=embedding_labels_nn,
                build_metrics=False)
    input_distribution = BinaryRNN(npd_config["hidden_dim"],)

    model = NeuralPolarDecoderOptimize(npd_const, npd_channel, input_distribution, channel)
    model.build(input_shape)

    if load_path is not None:
        try:
            model(tf.zeros(input_shape, dtype=tf.int32))
            model.load_weights(load_path)
            print(f"Loaded weights from {load_path}")
        except Exception as e:
            print(e)
            print(f"Model path {load_path} does not exist. Skipping loading weights.")
    return model