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
            print('Epoch {0} : {1} / {2}'.format(j, self.evaluate(test_data), n_test))
        else:
            print('Epoch {0} complete'.format(j))

    def update_mini_batch(self, mini_batch, eta):

        n_b = [np.zeros(b.shape) for b in self.biases]
        n_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            d_n_b, d_n_w = self.backprop(x, y)
            n_b = [nb + dnb for nb, dnb in zip(n_b, d_n_b)]
            n_w = [nw + dnw for nw, dnw in zip(n_w, d_n_w)]
        
        self.weights = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights,n_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.weights,n_b)]

    def backprop(self, x, y):

        n_b = [np.zeros(b.shape) for b in self.biases]
        n_w = [np.zeros(w.shape) for w in self.weights]

        #feed forward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        n_b[-1] = delta
        n_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = 





