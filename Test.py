import os
import numpy as np
import Util.CommonComponents.ActivationFunctions as af
import Util.CommonComponents.LoadData as ld


# Data File Path
rel_path_train_imgs = "python/Util/Data/MNIST/train-images-idx3-ubyte"
rel_path_train_lbls = "python/Util/Data/MNIST/train-labels-idx1-ubyte"
rel_path_test_imgs = "python/Util/Data/MNIST/t10k-images-idx3-ubyte"
rel_path_test_lbls = "python/Util/Data/MNIST/t10k-labels-idx1-ubyte"

# Load Data
train_imgs = ld.ReadMNISTBinaryImageFile(os.path.abspath(rel_path_train_imgs))
train_lbls = ld.ReadMNISTBinaryLabelFile(os.path.abspath(rel_path_train_lbls))
test_imgs = ld.ReadMNISTBinaryImageFile(os.path.abspath(rel_path_test_imgs))
test_lbls = ld.ReadMNISTBinaryLabelFile(os.path.abspath(rel_path_test_lbls))
print(np.shape(train_imgs))
print(np.shape(train_lbls))
print(np.shape(test_imgs))
print(np.shape(test_lbls))

# Hyper Parameters
## Size of Layers
input_layer_size = 784  # 28 * 28 
hidden_layer_size = 300
output_layer_size = 10
## Init Weights
w1 = np.random.rand(input_size, hidden_size)
w2 = np.random.rand(hidden_size, output_size)


