from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def preprocess_names(names, max_len=30):
    train_names = names
    train_names = [s.lower() for s in train_names]
    # tokenize names
    tokenizer = Tokenizer(char_level=True, lower=False)
    tokenizer.fit_on_texts(train_names)
    # encode input
    encoded_input = tokenizer.texts_to_sequences(train_names)
    # pad input
    return pad_sequences(encoded_input, maxlen=max_len, padding='post'), len(tokenizer.word_index) + 1
