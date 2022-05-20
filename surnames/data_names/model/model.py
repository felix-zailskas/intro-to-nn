import tensorflow as tf
import tensorflow.keras as keras


def create_rnn_model(hidden_inputs, nb_timesteps, nb_features, nb_classes, vocab_size, type='simple', bi_directional=False):
    rnn_layer = None
    if type == 'simple':
        rnn_layer = keras.layers.SimpleRNN(hidden_inputs, input_shape=(nb_timesteps, nb_features))
    if type == 'lstm':
        rnn_layer = keras.layers.LSTM(hidden_inputs, input_shape=(nb_timesteps, nb_features))
    if type == 'gru':
        rnn_layer = keras.layers.GRU(hidden_inputs, input_shape=(nb_timesteps, nb_features))
    if bi_directional:
        if type == 'simple':
            rnn_layer = keras.layers.Bidirectional(
                keras.layers.SimpleRNN(hidden_inputs),
                input_shape=(nb_timesteps, nb_features)
            )
        if type == 'lstm':
            rnn_layer = keras.layers.Bidirectional(
                keras.layers.LSTM(hidden_inputs),
                input_shape=(nb_timesteps, nb_features)
            )
        if type == 'gru':
            rnn_layer = keras.layers.Bidirectional(
                keras.layers.GRU(hidden_inputs),
                input_shape=(nb_timesteps, nb_features)
            )
    model = tf.keras.models.Sequential([
        keras.layers.Embedding(input_dim=vocab_size, output_dim=hidden_inputs, input_length=nb_timesteps),
        rnn_layer,
        keras.layers.Dense(nb_classes, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy'],
    )

    return model


def load_model(hidden_inputs, nb_timesteps, nb_features, nb_classes, vocab_size, type='simple', bi_directional=False):
    rnn_layer = None
    if type == 'simple':
        rnn_layer = keras.layers.SimpleRNN(hidden_inputs, input_shape=(nb_timesteps, nb_features))
    if type == 'lstm':
        rnn_layer = keras.layers.LSTM(hidden_inputs, input_shape=(nb_timesteps, nb_features))
    if type == 'gru':
        rnn_layer = keras.layers.GRU(hidden_inputs, input_shape=(nb_timesteps, nb_features))
    if bi_directional:
        if type == 'simple':
            rnn_layer = keras.layers.Bidirectional(
                keras.layers.SimpleRNN(hidden_inputs),
                input_shape=(nb_timesteps, nb_features)
            )
        if type == 'lstm':
            rnn_layer = keras.layers.Bidirectional(
                keras.layers.LSTM(hidden_inputs),
                input_shape=(nb_timesteps, nb_features)
            )
        if type == 'gru':
            rnn_layer = keras.layers.Bidirectional(
                keras.layers.GRU(hidden_inputs),
                input_shape=(nb_timesteps, nb_features)
            )
    model = tf.keras.models.Sequential([
        keras.layers.Embedding(input_dim=vocab_size, output_dim=hidden_inputs, input_length=nb_timesteps),
        rnn_layer,
        keras.layers.Dense(nb_classes, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy'],
    )

    checkpoint_path = f"checkpoints/cp_{'bi' if bi_directional else ''}{type}_{hidden_inputs}_2.ckpt"
    model.load_weights(checkpoint_path)

    return model
