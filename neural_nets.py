# Import numpy and random library
import numpy as np
import random

# Sigmoid and sigmoid prime 
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) / (1.0 - sigmoid(x))

# Create Network class
class Network (object):

    def __init__(self, sizes):

        self.num_layers = len(sizes) # sizes is a list of neurons in each layer
        self.sizes = sizes
        self.biases = [ np.random.randn(y, 1) for y in sizes[1:] ]
        self.weights = [ np.random.randn(y, x) 
                            for x, y in zip(sizes[:-1], sizes[1:]) ]
    
    # Define the forward steps
    def forward(self, a):

        # inner product with for loop
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        return a
    
    


