import random
import pickle
import numpy as np
import yaml
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense, Activation

vocab_dim = 15
embeding_dim = 100
input_length = 10
batch_size = 128
n_epoch = 200
hidden_num = 100
classes = 11
dropout = 0.5


def build_model():
    model = Sequential()
    model.add(Embedding(input_dim=vocab_dim, output_dim=embeding_dim, mask_zero=True, input_length=input_length))
    model.add(LSTM(hidden_num))
    model.add(Dropout(dropout))
    model.add(Dense(classes, activation='softmax'))
    model.add(Activation('softmax'))
    return model


def train_and_test(x_train, y_train, x_test, y_test):
    model = build_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=2)
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    yaml_string = model.to_yaml()
    with open('./model/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save('./model/lstm.h5')

    print('Test score:')
    print(model.metrics_names)
    print(score)


def process_data(split_ratio=0.1, max_length=10):
    path = './resource/data.txt'
    with open(path, 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    train_length = len(lines) * (1 - split_ratio)

    char2idx = {}
    idx2char = {}
    count = 1
    for k, line in enumerate(lines):
        label, sentence = line.rstrip('\n').split('\t')
        if len(sentence) < max_length:
            x_vector = [0] * max_length
            for i, char in enumerate(sentence):
                if char in char2idx:
                    x_vector[i] = char2idx[char]
                else:
                    char2idx[char] = count
                    x_vector[i] = count
                    idx2char[count] = char
                    count += 1

            y_vector = [0] * 11
            y_vector[int(label)] = 1
            if k < train_length:
                x_train.append(x_vector)
                y_train.append(y_vector)
            else:
                x_test.append(x_vector)
                y_test.append(y_vector)
    print(count)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    tmp = [char2idx, idx2char]
    with open('./resource/char_and_idx.pkl', 'wb') as f:
        pickle.dump(tmp, f)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = process_data()
    train_and_test(x_train, y_train, x_test, y_test)





