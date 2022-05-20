import os
import sys
import random
import numpy as np
from matplotlib import pyplot as plt
import plot.plotting
from model.model import create_rnn_model, load_model
import util.replicable_exp
import encoding.encoder
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def train_models(seq_length):
    # Loading the data from the csv file
    df_data = pd.read_csv("data/names_culture_origin.csv", sep="\t")

    # split test and train set
    padded_input, vocab_size = encoding.encoder.preprocess_names(df_data['Surname'].values, max_len=seq_length)

    df_data['Culture'] = df_data['Culture'].astype('category')
    nb_classes = df_data['Culture'].nunique()
    labels_onehot = to_categorical(df_data['Culture'].cat.codes)

    X_train, X_test, y_train, y_test = train_test_split(
        padded_input,
        labels_onehot,
        random_state=seed,
        shuffle=True,
        test_size=0.2
    )

    nb_timesteps = seq_length  # amount of inputs to process is the longest name in the input
    nb_features = 1  # amount of features per time-step is one character
    epochs = 1000
    hidden = [25, 50, 100, 500]
    methods = ["Simple", "LSTM", "Bi-directional LSTM"]
    types = ['simple', 'lstm', 'lstm']
    for j in range(len(types)):
        curr_type = types[j]
        curr_method = methods[j]
        for i in range(len(hidden)):
            hidden_dim = hidden[i]

            rnn = create_rnn_model(hidden_dim, nb_timesteps, nb_features, nb_classes, vocab_size, type=curr_type,
                                   bi_directional=j == len(types) - 1)

            checkpoint_path = f"checkpoints/cp_{'bi' if j == len(types) - 1 else ''}{curr_type}_{hidden_dim}_2.ckpt"
            # Create a callback that saves the model's weights
            cp_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,
                verbose=1
            )
            # Create a callback that stops training early to avoid over fitting
            cp_earlystop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=50
            )
            # train model
            train_history = rnn.fit(
                X_train,
                y_train,
                epochs=epochs,
                initial_epoch=0,
                verbose=1,
                callbacks=[cp_checkpoint, cp_earlystop],
                validation_split=0.2
            )
            plot.plotting.plot_loss(train_history,
                                    title=f"Loss for {curr_method} Structure with {hidden_dim} Hidden Input Size")
            plot.plotting.plot_acc(train_history,
                                   title=f"Accuracy for {curr_method} Structure with {hidden_dim} Hidden Input Size")
            loss, accuracy = rnn.evaluate(X_test, y_test, verbose=1)
            loss_train, accuracy_train = rnn.evaluate(X_train, y_train, verbose=1)
            print(f"Results for dim={hidden_dim}; type={curr_type}")
            print(f'Test set Accuracy {round(accuracy * 100, 2)}')
            print(f'Train set Accuracy {round(accuracy_train * 100, 2)}')
    plt.show()


def load_best_model():
    # Values from report and pre analysis
    return load_model(100, 20, 1, 6, 33, type='lstm', bi_directional=True)


if __name__ == '__main__':
    seed = 10
    seq_length = 20
    util.replicable_exp.set_seed(seed)
    if len(sys.argv) == 1:
        print("Please provide arguments")
    elif sys.argv[1] == 'Train':
        train_models(seq_length)
    else:
        model = load_best_model()
        df_predict = pd.read_csv("data/blind_names.csv")
        encoded, _ = encoding.encoder.preprocess_names(df_predict['Surname'], max_len=20)
        probabilities = model.predict(encoded)
        classes = ['American', 'Belgian', 'Dutch', 'Indian', 'Italian', 'Zimbabwean']
        origins = []
        for prob in probabilities:
            origins.append(classes[np.argmax(prob)])
        df_predict['origin'] = origins
        print(df_predict)
        df_predict.to_csv('prediction.csv', index=False)






