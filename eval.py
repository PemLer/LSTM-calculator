import pickle
import numpy as np
from keras.models import load_model

from utils.generate_data import generate_expression_v2
from utils.calculator import calculate


model_path = './model/lstm.h5'
model = load_model(model_path)
index_path = './resource/char_and_idx.pkl'
with open(index_path, 'rb') as f:
    char2idx, idx2char = pickle.load(f)


def eval_single(vector):
    """测试单条"""
    label = model.predict_classes(vector)
    return label


def sentence2vector(sentence, max_length=10):
    """将输入表达式转为向量"""
    vector = [0] * max_length
    for i, char in enumerate(sentence[:10]):
        if char in char2idx:
            vector[i] = char2idx[char]
        else:
            vector[i] = 0
    return np.array([vector])


if __name__ == '__main__':
    total = 0
    correct = 0
    while True:
        input()
        while True:
            expression = generate_expression_v2()
            try:
                res = calculate(expression)
            except ZeroDivisionError:
                continue
            if res < 0 or res > 10:
                continue
            vector = sentence2vector(expression)
            pre = eval_single(vector)
            tag = res == pre[0]
            print(expression, res, pre[0], tag)
            total += 1
            if tag:
                correct += 1
            print(total, "accuracy:", correct / total)
            break
