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
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, 
                            test_data = None):

        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)
            ]
        
        for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch, eta)

        if test_data:
            print('Epoch ')


