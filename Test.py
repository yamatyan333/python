import numpy as np
import pickle as pickle
import Util.CommonComponents.ActivationFunctions as af
import Util.CommonComponents.LoadData as ld
import matplotlib.pyplot as plt

########## Hyper Parameters ##########
## Size of Layers
input_layer_size = 784  # 28 * 28 
hidden_layer_size = 300
output_layer_size = 10

## Init Weights
w1 = np.random.rand(input_layer_size, hidden_layer_size)
w2 = np.random.rand(hidden_layer_size, output_layer_size)

## Batch Size
batch_size = 1000

## epoch num
epoch_number = 4000

## Learning Rate
learning_rate = 0.01

######################################

# Load Data
train_imgs = ld.ReadMNISTTrain_BinaryImageFile()
train_lbls = ld.ReadMNISTTrain_BinaryLabelFile()
test_imgs = ld.ReadMNISTTest_BinaryImageFile()
test_lbls = ld.ReadMNISTTest_BinaryLabelFile()
# print(np.shape(train_imgs))
# print(np.shape(train_lbls))
# print(np.shape(test_imgs))
# print(np.shape(test_lbls))

for epoch in range(0, epoch_number):
    # batch 0 -> 60,000 
    for batch_itr in range(batch_size, np.shape(train_lbls)[0] - 1, batch_size):
        print(batch_itr)

        # Mini Batch Data 
        x = (train_imgs[ batch_itr - batch_size : batch_itr - 1 ]) / 255
        t = train_lbls[ batch_itr - batch_size : batch_itr - 1 ] 
        t = np.identity(10)[t] # to One-Hot Label
        a = np.dot(x, w1) 
        z = af.SigmoidFunction(a)
        y = np.dot(z, w2)
        #plt.show(plt.imshow(np.reshape(x[0], (28,28))))
        #print(t[0])
        #exit()
        
        print("=========================")
        dw1 = np.dot(x.T, (z * (1 - z) * (np.dot((y-t), w2.T))))
        dw1 = np.sum(dw1, axis=0)/batch_size 
        print(np.shape(x) )
        print(np.shape(w1))
        #print(a)
        #print(z)
        print(y)
        #print(t)
        print("=========================")
        exit()
        # Update Hidden Layer Weight
        dw2 = np.dot(z.T, (y-t))
        dw2 = np.sum(dw2, axis=0)/batch_size
    
        w1 -= dw1 * learning_rate
        w2 -= dw2 * learning_rate
    





