import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, epochs=1000):
        """
        Initialize neural network with given layer sizes.
        
        Parameters:
        -----------
        layer_sizes : list
            List of integers specifying the number of neurons in each layer
        learning_rate : float
            Learning rate for gradient descent
        epochs : int
            Number of training epochs
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self.cost_history = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward_propagation(self, X):
        """
        Forward propagation step
        
        Parameters:
        -----------
        X : array-like
            Input features
            
        Returns:
        --------
        cache : list
            List of tuples containing activations and linear combinations
        """
        cache = []
        A = X
        
        # Forward propagate through each layer
        for i in range(len(self.weights)):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.sigmoid(Z)
            cache.append((A, Z))
            
        return cache
    
    def backward_propagation(self, X, y, cache):
        """
        Backward propagation step
        
        Parameters:
        -----------
        X : array-like
            Input features
        y : array-like
            Target values
        cache : list
            List of tuples containing activations and linear combinations
            
        Returns:
        --------
        gradients : dict
            Dictionary containing gradients for weights and biases
        """
        m = X.shape[0]
        gradients = {'weights': [], 'biases': []}
        
        # Calculate output layer gradients
        dA = cache[-1][0] - y.reshape(-1, 1)
        
        # Backpropagate through each layer
        for i in range(len(cache) - 1, -1, -1):
            Z = cache[i][1]
            if i == 0:
                A_prev = X
            else:
                A_prev = cache[i-1][0]
                
            dZ = dA * self.sigmoid_derivative(Z)
            dW = np.dot(A_prev.T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
                
            gradients['weights'].insert(0, dW)
            gradients['biases'].insert(0, db)
            
        return gradients
    
    def compute_cost(self, y_true, y_pred):
        """Calculate binary cross-entropy loss"""
        m = y_true.shape[0]
        cost = -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
        return cost
    
    def fit(self, X, y):
        """
        Train neural network using gradient descent
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Target values
        """
        for epoch in range(self.epochs):
            # Forward propagation
            cache = self.forward_propagation(X)
            
            # Compute cost
            cost = self.compute_cost(y.reshape(-1, 1), cache[-1][0])
            self.cost_history.append(cost)
            
            # Backward propagation
            gradients = self.backward_propagation(X, y, cache)
            
            # Update parameters
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * gradients['weights'][i]
                self.biases[i] -= self.learning_rate * gradients['biases'][i]
    
    def predict(self, X):
        """
        Make predictions for input features
        
        Parameters:
        -----------
        X : array-like
            Input features
            
        Returns:
        --------
        predictions : array-like
            Predicted probabilities
        """
        cache = self.forward_propagation(X)
        return cache[-1][0] 