import scipy as sp
import numpy as np
import Util.Common.ActivationFunctions as af
import Util.Common.LoadMnist as lm

def forwarding(x ,w1, h, w2):
    



dataset = lm.load_mnist()

# 層の深さ指定
input_size = 784
hidden_size = 100
output_size = 10

# 重み初期化
w1 = np.random.rand(input_size, hidden_size)
w2 = np.random.rand(hidden_size, output_size)

# 学習

# 入力層
input_Layer = np.random.rand(input_size)

# 中間層
hidden_layer = np.dot(input_Layer, w1)
hidden_layer = af.SigmoidFunction(hidden_layer)

# 出力層    
output_layer = np.dot(hidden_layer, w2)



