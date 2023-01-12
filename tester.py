from importdata import import_data
import numpy as np
import keras
import heapq
from sklearn.model_selection import train_test_split
import cmd

x, y, words, tokenizer = import_data()

model = keras.models.load_model("models/RNN_temp.h5")

seed_text = "Det finns många Journalister som"

train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio,
                                                        random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                    test_size=test_ratio / (test_ratio + validation_ratio),
                                                    random_state=42)


index_to_word = {}
for word, index in tokenizer.word_index.items():
    index_to_word[index] = word

# amount of sentances being tested
amount_of_words=25
pred_word = []
true_word = []

# Covert list of tokens back to words
def to_word(token_list):
    sentence = ""
    for token in token_list:
        sentence += index_to_word[token]+ " "

    return sentence

for i in range(amount_of_words):
    token_list = np.expand_dims(x_test[i], axis=0)
    prediction = model.predict(token_list, verbose=1)[0]
    pred_word.append(index_to_word[np.argmax(prediction)])
    true_word.append(index_to_word[y_test[i]])

header = ["input" "pred" "true"]

print(f'{"input":<50} {"pred":<15}{"true":<15}')
for i in range(amount_of_words):
    print(f'{to_word(x_test[i]):<50} {pred_word[i]:<15}{true_word[i]:<15}')

for x in range(11):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = np.expand_dims(token_list, axis=0)
    prediction = model.predict(token_list, verbose=1)[0]
    #prediction2 = index_to_word[np.argmax(prediction)]
    pred = heapq.nlargest(3, range(len(prediction)), key=prediction.__getitem__)
    pred2 = pred[0]-1
    prediction2 = index_to_word[pred2]
    seed_text += (" " + prediction2)
    print(seed_text)











