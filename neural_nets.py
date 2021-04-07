# Import numpy and random library
import numpy as np
import random

# Create Network class 
class Network (object):

    def __init__(self, sizes):

        self.num_layers = len(sizes) # sizes is a list of neurons in each layer
        self.sizes = sizes
        self.biases = [ np.random.randn(y, 1) for y in sizes[1:] ]
