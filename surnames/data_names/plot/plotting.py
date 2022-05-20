from matplotlib import pyplot as plt
import pandas as pd


def plot_acc(history, title=""):
    plt.figure()
    df = pd.DataFrame(history.history)
    plt.plot(df['accuracy'].rolling(10).mean())
    plt.plot(df['val_accuracy'].rolling(10).mean())
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')


def plot_loss(history, title=""):
    plt.figure()
    df = pd.DataFrame(history.history)
    plt.plot(df['loss'].rolling(10).mean())
    plt.plot(df['val_loss'].rolling(10).mean())
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
