import numpy as np

class SimpleNeuralNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2)
        return self.sigmoid(self.z2)
    
    def backward(self, X, y, output):
        # Hand-calculated gradients
        dW2 = np.dot(self.a1.T, (output - y) * self.sigmoid_derivative(self.z2))
        dW1 = np.dot(X.T, np.dot((output - y) * self.sigmoid_derivative(self.z2), 
                                 self.W2.T) * self.sigmoid_derivative(self.z1))
        return dW1, dW2
    
# Sample data 
X = np.array([[0, 1], [0, 2], [1, 3], [1, 4]])
y = np.array([[0], [1], [1], [0]])  #truth table

# Creating a simple network
nn = SimpleNeuralNet(input_size=2, hidden_size=2, output_size=1)

# Forward pass
output = nn.forward(X)

# Printing result
print("Input:")
print(X)
print("\nExpected Output:")
print(y.flatten())
print("\nActual Output (before training):")
print(output.flatten())
print("\nOutput Of each inpute:")
for i in range(len(X)):
    print(f"Input: {X[i]} - Expected: {y[i][0]} - Got: {output[i][0]:.4f}")