import numpy as np
import matplotlib.pyplot as plt

# Define the activation functions
def linear(x):
    
    return x

def sigmoid(x):
    
    return 1 / (1 + np.exp(-x))

def tanh(x):
    
    return np.tanh(x)

def relu(x):
    
    return np.maximum(0, x)

def softmax(x):
    
    # Subtracting the maximum value for numerical stability
    exp_scores = np.exp(x - np.max(x))  
    return exp_scores / np.sum(exp_scores)

def leaky_relu(x, alpha=0.01):
    
    return np.where(x > 0, x, alpha * x)

def swish(x):
    
    return x * sigmoid(x)


# Generate input data
x = np.linspace(-5, 5, 100)

# Calculate the outputs of each activation function

outputs = {
    'Linear': linear(x),
    'Sigmoid': sigmoid(x),
    'Tanh': tanh(x),
    'ReLU': relu(x),
    'Softmax': softmax(x),
    'Leaky ReLU': leaky_relu(x),
    'Swish': swish(x)
}

# Calculate the sum of the absolute values of outputs for normalization
abs_sum = sum(np.abs(val).sum() for val in outputs.values())

# Plot pie chart
plt.figure(figsize=(10, 8))
plt.pie([np.abs(val).sum() / abs_sum for val in outputs.values()], labels=outputs.keys(), autopct='%1.1f%%', startangle=140)
plt.title('Relative Magnitudes of Activation Functions')
plt.show()
