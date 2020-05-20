"""
中缀表达式求值
支持 + - * / ( )
"""


def calculate(expression):
    nums = []
    ops = []

    def cal():
        val1 = nums.pop()
        val2 = nums.pop()
        op = ops.pop()
        if op == '+':
            nums.append(val2 + val1)
        elif op == '-':
            nums.append(val2 - val1)
        elif op == '*':
            nums.append(val2 * val1)
        elif op == '/':
            nums.append(val2 // val1)

    expression = '(' + expression + ')'
    i = 0
    while i < len(expression):
        if expression[i] == ' ':
            i += 1
            continue
        elif expression[i].isdigit():
            tmp = ''
            while i < len(expression) and expression[i].isdigit():
                tmp += expression[i]
                i += 1
            nums.append(int(tmp))
        elif expression[i] == '(':
            ops.append(expression[i])
            i += 1
        elif expression[i] == ')':
            while ops[-1] != '(':
                cal()
            ops.pop()
            i += 1
        elif expression[i] == '*' or expression[i] == '/':
            while ops[-1] == '*' or ops[-1] == '/':
                cal()
            ops.append(expression[i])
            i += 1
        elif expression[i] == '+' or expression[i] == '-':
            while ops[-1] != '(':
                cal()
            ops.append(expression[i])
            i += 1
    return nums[-1]


if __name__ == '__main__':
    exps = [
        '(2 +1)*3',
        '9-3*7+4',
        '9/(6)'
    ]
    for exp in exps:
        print(exp, '=', calculate(exp))
