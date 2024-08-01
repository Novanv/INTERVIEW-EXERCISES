import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the Neural Network with random weights and biases.
        
        Parameters:
        - input_size (int): The number of input features.
        - hidden_size (int): The number of neurons in the hidden layer.
        - output_size (int): The number of output neurons (classes).
        """
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01  # Weight matrix for input to hidden layer
        self.b1 = np.zeros((1, hidden_size))  # Bias for hidden layer
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01  # Weight matrix for hidden to output layer
        self.b2 = np.zeros((1, output_size))  # Bias for output layer

    def forward(self, X):
        """
        Performs a forward pass through the network.
        
        Parameters:
        - X (numpy.ndarray): Input data of shape (number of examples, input_size).
        
        Returns:
        - numpy.ndarray: Output of the network of shape (number of examples, output_size).
        """
        self.Z1 = np.dot(X, self.W1) + self.b1  # Linear transformation for hidden layer
        self.A1 = np.maximum(0, self.Z1)  # ReLU activation function
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # Linear transformation for output layer
        return self.Z2

    def backward(self, X, dZ2, learning_rate):
        """
        Performs a backward pass to compute gradients and update weights and biases.
        
        Parameters:
        - X (numpy.ndarray): Input data of shape (number of examples, input_size).
        - dZ2 (numpy.ndarray): Gradient of the loss with respect to the output of the network.
        - learning_rate (float): Learning rate for updating the weights and biases.
        """
        m = X.shape[0]  # Number of examples

        # Compute gradients for W2 and b2
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Compute gradients for hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0)  # ReLU derivative
        
        # Compute gradients for W1 and b1
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2