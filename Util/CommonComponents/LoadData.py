import os
import struct
import numpy as np
import matplotlib.pyplot as plt

# Data File Path
path_train_imgs = os.path.abspath("~/python/Util/Data/MNIST/train-images-idx3-ubyte")
path_train_lbls = os.path.abspath("~/python/Util/Data/MNIST/train-labels-idx1-ubyte")
path_test_imgs = os.path.abspath("~/python/Util/Data/MNIST/t10k-images-idx3-ubyte")
path_test_lbls = os.path.abspath("~/python/Util/Data/MNIST/t10k-labels-idx1-ubyte")

def __ReadMNISTBinaryImageFile(path):

    # Return value
    imgArray = np.empty
    
    # Read file
    print("Now, MNIST Binary Images are being loaded ...")
    with open(path, "rb") as f:
        # Extract Header
        magicNumber, images, rows, cols = struct.unpack('>iiii', f.read(16))
        print(" ===== IMAGE DATA ===== ")
        print("MagicNum :" , magicNumber)
        print("images :" , images)
        print("rows :" , rows)
        print("cols :" , cols)
        print(" ====================== ")
    
        # For Check the First Image
        #value = np.array(struct.unpack('>784B', f.read(rows*cols)))
        #value = np.reshape(value, (rows, cols))
        #print(type(value))
        #print(value.shape)
        #plt.imshow(value, cmap='gray', vmin=0, vmax=255)
        #plt.show()
    
        # Read All Data
        imgArray = np.array(struct.unpack('>' + str(images*rows*cols) + 'B', f.read()))
        imgArray = np.reshape(imgArray, (images, rows*cols))
    print("MNIST Binary Images are loaded successfully!\n")
    # Reshape Data
    #imgArray = np.reshape(imgArray, (10000, 28, 28))
    #plt.imshow(imgArray[0], cmap='gray', vmin=0, vmax=255)
    #plt.show()
    return imgArray


def __ReadMNISTBinaryLabelFile(path):
    lblArray = np.empty

    # Read file
    print("Now, MNIST Binary Labels are being loaded ...")
    with open(path, "rb") as f:
        # Extract Header
        magicNumber, labels = struct.unpack('>ii', f.read(8))
        print(" ===== LABEL DATA ===== ")
        print("MagicNum :" , magicNumber)
        print("labels :" , labels)
        print(" ====================== ")
    
        # For Check the First Image
        #value = np.array(struct.unpack('>5B', f.read(5)))
        #print(type(value))
        #print(value.shape)
        #print(value)
    
        # Read All Data
        lblArray = np.array(struct.unpack('>' + str(labels) + 'B', f.read()))
    print("MNIST Binary Labels are loaded successfully!\n")
    return lblArray


def ReadMNISTTrain_BinaryImageFile():
    return __ReadMNISTBinaryImageFile(path_train_imgs)

def ReadMNISTTrain_BinaryLabelFile():
    return __ReadMNISTBinaryLabelFile(path_train_lbls)

def ReadMNISTTest_BinaryImageFile():
    return __ReadMNISTBinaryImageFile(path_test_imgs)

def ReadMNISTTest_BinaryLabelFile():
    return __ReadMNISTBinaryLabelFile(path_test_lbls)
