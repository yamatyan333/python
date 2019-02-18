import numpy

def SigmoidFunction(x):
    return 1 / (1 + numpy.exp(-x))
