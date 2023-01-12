
import re
import numpy as np

import csv
from tensorflow.keras.preprocessing.text import Tokenizer


def import_data():
    # Use a breakpoint in the code line below to debug your script.

    with open('text-ds.csv', 'r', encoding="utf8") as file:
        text = file.read()

    # removes unnecessary html and symbols
    clean_text = re.sub('<.*?>|\(.*?\)|:|,|;|\?|“|”|"|!|►|\xa0|\u200b|\u2009|\xad|\x7f', ' ', text)


    # writes the cleaned text to a textfile for inspection to find issues
    # f = open("clean_text.txt", "w", encoding='utf8')
    # f.write(clean_text)
    # f.close()

    # turn all the data into lowercase
    corpus = clean_text.lower().split('\n',)
    print(len(corpus))
    print(type(corpus))
    print(corpus[:2])


    # tokenizing the words,
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    word_index = tokenizer.word_index
    total_unique_words = len(tokenizer.word_index)+1

    f = open('word_dictionary3.csv', 'w', encoding='utf-8', newline="")
    writer = csv.writer(f)
    for index, x in enumerate(word_index):
        writer.writerow([index, x])
    f.close()
    # print(word_index)
    print("amount of unique words: ",total_unique_words)

    # creating n grams of the words
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        i = 0
        while i+5 < len(token_list):
            n_gram_seqs = token_list[i:i+6]
            input_sequences.append(n_gram_seqs)
            i += 1


    print("input seqs", len(input_sequences))
    print(input_sequences[:2])


    # split data into features and labels
    # last word in a sequence is the label
    input_sequences = np.array(input_sequences)
    x_values, labels = input_sequences[:, :-1], input_sequences[:, -1]
    #y_values = tf.keras.utils.to_categorical(labels, num_classes= total_unique_words)

    print(x_values[:3])
    print(labels[:3])

    return x_values, labels, total_unique_words, tokenizer
