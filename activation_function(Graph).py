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

# Apply each activation function to the input data
linear_output = linear(x)
sigmoid_output = sigmoid(x)
tanh_output = tanh(x)
relu_output = relu(x)
softmax_output = softmax(x)
leaky_relu_output = leaky_relu(x)
swish_output = swish(x)

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(x, linear_output, label='Linear', linewidth=2)
plt.plot(x, sigmoid_output, label='Sigmoid', linewidth=2)
plt.plot(x, tanh_output, label='Tanh', linewidth=2)
plt.plot(x, relu_output, label='ReLU', linewidth=2)
plt.plot(x, softmax_output, label='Softmax', linewidth=2)
plt.plot(x, leaky_relu_output, label='Leaky ReLU', linewidth=2)
plt.plot(x, swish_output, label='Swish', linewidth=2)

plt.title('Activation Functions')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
