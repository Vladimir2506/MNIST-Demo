# PrepareData.py
# Load dataset from file

from tensorflow.examples.tutorials.mnist import input_data

def LoadDataset():
    dataset_mnist = input_data.read_data_sets('F:\\MNIST', one_hot = True)
    return dataset_mnist

