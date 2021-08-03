import numpy as np

class NeuralNetwork:
    
    def __init__(self, layers, alpha=0.1):
        self.W = []
        self.layers = layers
        self.alpha = alpha
        
        # Loop over the layers except the last two
        for i in np.arange(0, len(layers) - 2):
            # Randomly init weight matrix and add an extra node for bias term
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))
            
        # Input connection needs a biass term, output doesn't
        w = np.random.randn(layers[-2] + 1, layers [-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        return "Neural Network: {}".format("-".join(str(l) for l in self.layers))
    
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))
    
    def sigmoid_deriv(self, x):
        return x * (1 - x)
    
    def fit(self, X, y, epochs=100, display_update=100):
        # Add the bias term at the end of the matrix
        X = np.c_[X, np.ones((X.shape[0]))]
        
        for epoch in np.arange(0, epochs):
            for x, target in zip(X, y):
                self.fit_partial(x, target)
                
            if epoch == 0 or (epoch + 1) % display_update == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] Epoch={}, loss={:.7f}".format(epoch + 1, loss))
                
    def fit_partial(self, x, y):
        A = [np.atleast_2d(x)]

        # Feed Forward
        # Loop over the layers in the network
        for layer in np.arange(0, len(self.W)):
            # Calculate the "net input" to the current layer by taking the dot
            # product of activation and weight matrix
            net = A[layer].dot(self.W[layer])
            
            # Calculate the "net output"
            out = self.sigmoid(net)
            
            # Add net output to activation.
            # Final entry in activation is the output of the last layer in the network
            A.append(out)
            
        # Back Propogation
        # Compute the difference between prediction and the ground truth
        error = A[-1] - y
        
        # Build list of deltas using the chain rule
        # The first entry is the error of the output layer times the derivative
        # of activation function for the output value
        D = [error * self.sigmoid_deriv(A[-1])]
        
        # Ignore the last two because we have already taken them in account
        for layer in np.arange(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)
            
        # Since deltas are in reverse order, reverse deltas
        D = D[::-1]
        
        # Weight Update Phase
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])
            
    def predict(self, X, add_bias=True):
        p = np.atleast_2d(X)
        
        # Check if bias term needs to be added in the matrix
        if add_bias:
            p = np.c_[p, np.ones((p.shape[0]))]
            
        for layer in np.arange(0, len(self.W)):
            # Compute the output prediction
            p = self.sigmoid(np.dot(p, self.W[layer]))
            
        return p

    def calculate_loss(self, X, y):
        targets = np.atleast_2d(y)
        predictions = self.predict(X, add_bias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        
        return loss
