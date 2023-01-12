from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping

def build_model(words):
    model = Sequential()

    model.add(Embedding(input_dim=words,
                        output_dim=100,
                        weights=None,
                        trainable=True))

    #model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
    model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0,activation='tanh',recurrent_activation='sigmoid', unroll=False, use_bias=True))

    model.add(Dense(64,activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(words, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def generate_callback(patience):
    checkpoint = ModelCheckpoint("models/RNN_temp.h5",
                                 monitor="val_loss",
                                 mode="min",
                                 save_best_only=True,
                                 verbose=1)

    # EarlyStopping to find best model with a large number of epochs
    earlystop = EarlyStopping(monitor='val_loss',
                              restore_best_weights=True,
                              patience=patience,  # number of epochs with
                              # no improvement after which
                              # training will be stopped
                              verbose=1)

    return [checkpoint, earlystop]