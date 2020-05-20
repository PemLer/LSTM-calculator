"""
10以内的数
元算符号：+，-，*，/，（，）
长度：3-10
"""
import random
from utils.calculator import calculate


def generate_expression_v2():
    """
    10000
    总数：(10^5 * 4^4) * (1/40) ≈ 640000
    :return:
    """
    nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    ops = ['+', '-', '*', '/']
    exp = []
    for i in range(4):
        exp.append(random.choice(nums))
        exp.append(random.choice(ops))
    exp.append(random.choice(nums))
    return ''.join(exp)


if __name__ == '__main__':
    count = 40000
    data = []
    while count > 0:
        expression = generate_expression_v2()
        try:
            res = calculate(expression)
        except ZeroDivisionError:
            continue
        if 0 <= res <= 10:
            count -= 1
            data.append([expression, res])

    file_path = '../resource/data.txt'
    with open(file_path, 'w') as f:
        for expression, res in data:
            line = '{}\t{}\n'.format(res, expression)
            f.write(line)
