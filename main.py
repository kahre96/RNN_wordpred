from importdata import import_data
from RNN_model import build_model
from RNN_datagen import RNNDataGenerator
from sklearn.model_selection import train_test_split
from RNN_model import generate_callback

if __name__ == '__main__':
    x, y, words, tokenizer = import_data()


    # variabels to determine the ratio of the split data
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio,
                                                        random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                    test_size=test_ratio / (test_ratio + validation_ratio),
                                                    random_state=42)

    model = build_model(words)

    generator = RNNDataGenerator(x_train, y_train, 256,words)
    val_generator = RNNDataGenerator(x_val, y_val,256,words)
    test_generator = RNNDataGenerator(x_test, y_test,256,words)

    callbacks = generate_callback(5)

    history = model.fit(generator,
                        epochs=50,
                        validation_data=val_generator,
                        verbose=1,
                        batch_size=256,
                        callbacks=callbacks)

    model.evaluate(test_generator)







    model.save("models/50e1s.h5")



