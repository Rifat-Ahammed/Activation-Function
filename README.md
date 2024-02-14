# Activation-Function for Deep Learning (Neural network).
### Assignment_01(Rifat Ahammed): Understanding and implementing the activation function.<br/>
Throughout this article we will learn about Activation Functions with real life example and also get answer that why they are needed and what their types.

### Objective:
1. To comprehend the conceptual and mathematics underpinnings of the Activation Function.
2. To execute the Activation Function in a programming language (such as Python).
3. The objective is to examine the attributes and consequences of using the Activation Function
inside neural networks.<br /> 

## 1. Theoretical Understanding:
  ### o Explain the Activation Function, including its equation and graph.<br/>
  ### Activation fucntion: 
  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/image.png)<br/>
  An activation function in a neural network is a mathematical function applied to the output of a neuron or a layer of neurons. It determines the output or activation level of a neuron based on the weighted sum of its inputs.<br/>
  The purpose of an activation function is to introduce non-linear transformations to the network’s computations. Without activation functions, the network would be limited to performing only linear transformations.<br/>
  
  ### Equation:
  An activation function is typically represented by a function f(x) where x is the input
  to the neuron. Different activation functions have different equations. One commonly
  used activation function is the sigmoid function: 
  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/image-1.png)

  ### Graph:
  The sigmoid function graph is an S-shaped curve that smoothly transitions between 0 and 1 as the input varies from negative to positive infinity:
  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/image-2.png)
  <br/>

  ### o Discuss why activation functions are used in neural networks, focusing on the role of the Activation function.<br/>
  
  ### Activation functions are used in neural networks for several reasons:<br/>
  
  1. **Introducing non-linearity:** Without activation functions, neural networks would simply be a series of linear transformations. However, many real-world problems are inherently non-linear, and thus require non-linear transformations to be effectively modeled. Activation functions introduce non-linearity into the output of each neuron, allowing neural networks to model more complex relationships between inputs and outputs.<br/>
  2. **Stabilizing gradients:** When training a neural network using backpropagation, the gradients can become unstable and either vanish or explode. Activation functions can help to stabilize the gradients and make training more efficient.<br/>
  3. **Providing output range:** Activation functions can restrict the output of a neuron to a certain range, such as between 0 and 1 for the sigmoid function or between -1 and 1 for the tanh function. This can be useful for certain types of problems, such as binary classification or regression with outputs bounded by certain limits.<br/>
  4. **Non-monotonic functions:**  Certain activation functions are non-monotonic, which means that they introduce local maxima and minima in the output of the neuron. This can help to prevent the network from getting stuck in local optima during training and improve its ability to find the global optimum.<br/>


## 2. Mathematical Exploration:<br/>

  ### o Derive the Activation function formula and demonstrate its output range.<br/>
  The primary role of the Activation Function is to transform the summed weighted input from the node into an output value to be fed to the next hidden layer or as output. 
  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/image-3.png)<br/>
  ### Elements of a Neural Networks Architecture:
  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/image-4.png)<br/>
  In the image above, you can see a neural network made of interconnected neurons. Each of them is characterized by its weight, bias, and activation function.<br/>

  **Here are other elements of this network.**<br/>

  **1. Input Layer:** The input layer takes raw input from the domain. No computation is performed at this layer. Nodes here just pass on the information (features) to the hidden layer.<br/>

  **2. Hidden Layer:** As the name suggests, the nodes of this layer are not exposed. They provide an abstraction to the neural network. The hidden layer performs all kinds of computation on the features entered through the input layer and transfers the result to the output layer.<br/>

  **3. Output Layer:** It’s the final layer of the network that brings the information learned through the hidden layer and delivers the final value as a result.<br/>

  📢 **Note:** All hidden layers usually use the same activation function. However, the output layer will typically use a different activation function from the hidden layers. The choice depends on the goal or type of prediction made by the model.

  **Mathematical Exploration:**<br/>
  **let’s understand what these derivatives are and how to calculate them:**<br/>
  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/image-5.png)<br/>

  **1. Input layer**:<br/>
    Now from the image above, we can take raw input.Consider a neural network with a single neuron layer. Each neuron in this layer takes inputs i.e. i1, i2,...,in and produces an output z.

  **Hidden layer (layer 1):** <br/>

    z(1) = W(1)X + b(1) a(1)
    Here,
    z(1) is the vectorized output of layer 1
    W(1) be the vectorized weight assigned to neurons of hidden layer i.e. w1, w2, w3 and w4
    X be the vectorized input features i.e. i1 and i2
    b is the vectorized bias assigned to neurons in hidden layer i.e. b1 and b2
    a(1) is the vectorized form of any linear function.<br/>

  **Output layer (layer 2):**<br/>

  Note : Input for layer 2 is output from layer 1.<br/>
  
    z(2) = W(2)a(1) + b(2) 
    a(2) = z(2)

**Calculation at Output layer:**<br/>

    z(2) = (W(2) * [W(1)X + b(1)]) + b(2)
    z(2) = [W(2) * W(1)] * X + [W(2)*b(1) + b(2)]

Let, 

    [W(2) * W(1)] = W
    [W(2)*b(1) + b(2)] = b

    Final output : z(2) = W*X + b
    which is again a linear function

  o Calculate the derivative of the Activation function and explain its significance in the backpropagation process.<br/>
  ### Different kind of activation functions and their output ranges:

  <br/>There are several commonly used activation functions in neural networks, including:<br/>

  **1. Linear Function:**

  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/image6.png)<br/>

  **Equation:** Linear function has the equation similar to as of a straight line i.e. 
    
    y = x

  No matter how many layers we have, if all are linear in nature, the final activation function of last layer is nothing but just a linear function of the input of first layer.<br/>

  **Range:** 
    
    -inf to +inf
    
  **Uses:** Linear activation function is used at just one place i.e. output layer.<br/>

  **Issues:** If we will differentiate linear function to bring non-linearity, result will no more depend on input “x” and function will become constant, it won’t introduce any ground-breaking behavior to our algorithm.<br/>



  2. **Sigmoid Function:** 
  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/image7.png)<br/>

  It is a function which is plotted as ‘S’ shaped graph.<br/>
  **Equation:** 
    
    A = 1/(1 + e^-x)<br/>

  **Nature:** Non-linear. Notice that X values lies between -2 to 2, Y values are very steep. This means, small changes in x would also bring about large changes in the value of Y.<br/>

  **Range:** 
    
    0 to 1

  **Uses:** Usually used in output layer of a binary classification, where result is either 0 or 1, as value for sigmoid function lies between 0 and 1 only so, result can be predicted easily to be 1 if value is greater than 0.5 and 0 otherwise.<br/>

  **3. Tanh Function:**
  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/image8.png)<br/>

  The activation that works almost always better than sigmoid function is Tanh function also known as Tangent Hyperbolic function. It’s actually mathematically shifted version of the sigmoid function. Both are similar and can be derived from each other.

  **Equation:**

    f(x) = tanh(x) = 2/(1+e^(-2x))-1

  **Range :** 
  
    -1 to +1

  **Nature :** non-linear<br/>

  **Uses :** Usually used in hidden layers of a neural network as it’s values lies between -1 to 1 hence the mean for the hidden layer comes out be 0 or very close to it, hence helps in centering the data by bringing mean close to 0. This makes learning for the next layer much easier.<br/>


  **4. ReLU Function:**<br/>
  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/image9.png)<br/>

  It Stands for Rectified linear unit. It is the most widely used activation function. Chiefly implemented in hidden layers of Neural network.

  **Equation :**
    
     A(x) = max(0,x). 
     It gives an output x if x is positive and 0 otherwise.

  **Range :**
    
     [0, inf)

  **Nature :** non-linear, which means we can easily backpropagate the errors and have multiple layers of neurons being activated by the ReLU function.<br/>

  **Uses :** ReLu is less computationally expensive than tanh and sigmoid because it involves simpler mathematical operations. At a time only a few neurons are activated making the network sparse making it efficient and easy for computation.<br/>

  **In simple words, RELU learns much faster than sigmoid and Tanh function.**
  
  4. **Softmax:** Softmax activation is typically used as the final activation function in a neural network for multiclass classification problems. It maps its inputs to a probability distribution over multiple classes.<br/>
  5. **Leaky ReLU:** It is similar to the ReLU function but allows a small gradient for negative inputs, preventing neurons from dying (example: outputting zero for all inputs).<br/>
  6. **Swish:** Swish is a recent activation function that has been shown to outperform ReLU on some tasks. It is defined as x * sigmoid(x).<br/>


<br/>


