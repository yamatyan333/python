import numpy as np

def SigmoidFunction(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(a):
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y
