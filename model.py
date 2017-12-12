from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential

def model_fn(
        config,
        input_length):

    m = Sequential()
    m.add(Embedding(input_length, config.char_embed_size))
    m.add(LSTM(config.hidden_sizes[0], return_sequences=True, dropout=config.input_dropout,
               recurrent_dropout=config.input_dropout))
    m.add(LSTM(config.hidden_sizes[1], return_sequences=True, dropout=config.dropout,
               recurrent_dropout=config.dropout))
    m.add(LSTM(config.hidden_sizes[2], dropout=config.dropout, recurrent_dropout=config.dropout))
    m.add(Dense(config.label_size, activation=config.activation))
    m.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    return m
