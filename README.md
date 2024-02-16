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



  **2. Sigmoid Function:** 
  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/image7.png)<br/>

  It is a function which is plotted as ‘S’ shaped graph.<br/>
  **Equation:** 
    
    A = 1/(1 + e^-x)<br/>

  **Nature:** Non-linear. Notice that X values lies between -2 to 2, Y values are very steep. This means, small changes in x would also bring about large changes in the value of Y.<br/>

  **Range:** 
    
    0 to 1

  **Uses:** Usually used in output layer of a binary classification, where result is either 0 or 1, as value for sigmoid function lies between 0 and 1 only so, result can be predicted easily to be 1 if value is greater than 0.5 and 0 otherwise.<br/>

  **3. Tanh Function:**
  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/tanh.PNG)<br/>

  The activation that works almost always better than sigmoid function is Tanh function also known as Tangent Hyperbolic function. It’s actually mathematically shifted version of the sigmoid function. Both are similar and can be derived from each other.

  **Equation:**

    f(x) = tanh(x) = 2/(1+e^(-2x))-1

  **Range :** 
  
    -1 to +1

  **Nature :** Non-linear<br/>

  **Uses :** Usually used in hidden layers of a neural network as it’s values lies between -1 to 1 hence the mean for the hidden layer comes out be 0 or very close to it, hence helps in centering the data by bringing mean close to 0. This makes learning for the next layer much easier.<br/>


  **4. ReLU Function:**<br/>
  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/image9.png)<br/>

  It Stands for Rectified linear unit. It is the most widely used activation function. Chiefly implemented in hidden layers of Neural network.

  **Equation :**
    
     A(x) = max(0,x). 
     It gives an output x if x is positive and 0 otherwise.

  **Range :**
    
     [0, inf)

  **Nature :** Non-linear, which means we can easily backpropagate the errors and have multiple layers of neurons being activated by the ReLU function.<br/>

  **Uses :** ReLu is less computationally expensive than tanh and sigmoid because it involves simpler mathematical operations. At a time only a few neurons are activated making the network sparse making it efficient and easy for computation.<br/>

  **In simple words, RELU learns much faster than sigmoid and Tanh function.**
  
  **5. Softmax Function:**<br/>
  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/Softmax.png)<br/>

  The softmax function is also a type of sigmoid function but is handy when we are trying to handle multi- class classification problems.<br/>

  **Equation :**
  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/softmax1.PNG)<br/>


  **Nature :** Non-linear<br/>

  **Uses :** Usually used when trying to handle multiple classes. the softmax function was commonly found in the output layer of image classification problems.The softmax function would squeeze the outputs for each class between 0 and 1 and would also divide by the sum of the outputs.<br/> 

  **Output:** The softmax function is ideally used in the output layer of the classifier where we are actually trying to attain the probabilities to define the class of each input.<br/>

  **6. Leaky ReLU:** 
  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/leaky_RELU.png)<br/>

  A type of activation function based on a ReLU, but it has a small slope for negative values instead of a flat slope. The slope coefficient is determined before training, i.e. it is not learnt during training.<br/>

  **Equation:**

      f(x)=max(0.01*x , x)
  
  **Nature :** linear<br/>

  **Range:**

      (-inf to inf)

  **Uses:** Leaky ReLU is an activation function used in artificial neural networks to introduce nonlinearity among the outputs between layers of a neural network.<br/>

  **7. Swish Function:**
  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/swish.png)<br/>

  This function uses non-monotonicity, and may have influenced the proposal of other activation functions with this property such as Mish<br/>
  **Equation:**

    f(x)=x⋅sigmoid(βx) = x/(1+e^-βx) 
    where β is a constant or trainable parameter.
  
  **Range:**

    (0, 1)

  **Nature:** Non Linear.<br/>

  **Uses:** Swish activation is used in the Long Short-Term Memory (LTST) neural networks, which are used extensively in sequence prediction and likelihood problems.<br/>


  **Note:**

    1. The basic rule of thumb is if you really don’t know what activation function to use, then simply use RELU as it is a general activation function in hidden layers and is used in most cases these days.

    2. If your output is for binary classification then, sigmoid function is very natural choice for output layer.

    3. If your output is for multi-class classification then, Softmax is very useful to predict the probabilities of each classes. 
  
  <br/>

  **o Calculate the derivative of the Activation function and explain its significance in the backpropagation process.**
  <br/>

  **1. Linear Activation Function:**<br/>

  The linear activation function f(x) is defined as:

    f(x)=x

  **Derivative of Linear Activation Function:**<br/>

  Since the linear activation function is simply f(x)=x, its derivative is a constant, which is always 1. Therefore, the derivative of the linear activation function 

    f′(x)=1

 **2. Sigmoid Function:**<br/>

 The sigmoid function, also known as the logistic function, is a common activation function used in neural networks, particularly in binary classification tasks. It maps any real-valued number to the range [0, 1]. The sigmoid function is defined as:

  $\large{f(x)}= \frac{1}{(1+e^{−x})}$

  Where e is the base of the natural logarithm (approximately equal to 2.71828).<br/>

  **Derivative of Sigmoid Function:**<br/>

  The derivative of the sigmoid function f′(x) can be derived as follows:<br/>

   <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/sig_1.PNG)<br/>

   So, the derivative of the sigmoid function f ′(x) is:<br/>

  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/sig_4.PNG)<br/>

  **Tanh Function:**<br/>

  The hyperbolic tangent (tanh) activation function is a common non-linear activation function used in neural networks. It squashes the input values to the range [-1, 1]. It's defined as: 

  $\large{Tanh(x) = f(x)}= \frac{e^x-e^{-x}}{e^x+e^{−x}}$<br/>

  Using **quotient rule** and **Euler’s identities** we can simplify it to:
  
  $\large{f(x)}= 1-(\frac{e^x-e^{-x}}{e^x+e^{−x}})^2$<br/>
  <br/>
  **Applying Euler’s identities:<br/>**

  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/tanh_1.PNG)<br/>

  Applying quotient rule to every term above, we have<br/>

  ![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/tanh_2.PNG)<br/>

  Putting all together we have

  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/tanh_3.PNG)<br/>
  
  **ReLU Function:**<br/>

  The Rectified Linear Unit (ReLU) activation function is one of the most commonly used activation functions in neural networks. It's defined as:<br/>

    f(x)=max(0,x)

  **Derivative of ReLU Function:**<br/>

  The ReLU function is piecewise linear, so its derivative is:<br/>
  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/relu.PNG)<br/>
 
 **Softmax Function:**<br/>

  The softmax function is commonly used in the output layer of a neural network for multi-class classification tasks. It takes a vector of real-valued scores (often called logits) as input and outputs a probability distribution over multiple classes. The softmax function is defined as follows:

  Given a vector z of length K (where K is the number of classes), the softmax function computes the probabilities 
  $p_i$
  for each class i as:

  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/sotmax.PNG)<br/>

  Where:

  1. e is the base of the natural logarithm (Euler's number).
  2. $(z_i)$ is the i-th element of the input vector z.
  3. The denominator is the sum of exponentials of all elements in the input vector, which ensures that the output is a valid probability distribution (i.e., the probabilities sum up to 1).

  Here's the simplified expression for the gradient of the cross-entropy loss with respect to the logits, which effectively incorporates the derivative of the softmax function:<br/>

  $\large\frac{∂Loss}{∂z_i}= \vec{y_i}−y_i$<br/>

  Where:<br/>

  1. $\vec{y_i}$ is the predicted probability (output of the softmax function) for class *i*.

  2. ${y_i}$ is the true label (one-hot encoded) for class *i*.<br/>
  3. ${z_i}$ is the input logit for class *i*.

  <br/>

  **Leaky ReLU:**
  The Leaky ReLU activation function is a variant of the traditional ReLU function. It addresses the issue of "dying ReLU" neurons that can occur when the input to a ReLU neuron is consistently negative, causing the neuron to always output zero.

  The Leaky ReLU function is defined as follows:

  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/L_ReLU.PNG)<br/>

  Where α is a small constant (typically around 0.01) called the "leak" coefficient.

  **Derivative of Leaky ReLU Function:**

  The derivative of the Leaky ReLU functionf′(x) is computed as follows:

  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/R_ReLU1.PNG)<br/>

  **Swish Function:**<br/>

  The Swish activation function was proposed by researchers at Google in 2017 as an alternative to other activation functions like ReLU. It is defined as:

      f(x)=x⋅sigmoid(x)

  Where sigmoid(x) is the sigmoid function:

  $\large{sigmoid(x)} = \frac{1}{1+e^{-x}}$
  <br/>

  **Derivative of Swish Function:**
  <br/>

  To find the derivative f′(x) of the Swish function, we can use the product rule:
    
    f′(x) = (x⋅sigmoid(x))′
          = x′⋅sigmoid(x) + x⋅sigmoid′(x)
          = sigmoid(x) + x⋅sigmoid′(x)
          = sigmoid(x) + x⋅sigmoid(x)⋅(1−sigmoid(x))
          = sigmoid(x) + x⋅f(x)⋅(1−sigmoid(x))
​
 
  Where, f(x) is the output of the Swish function.
  <br/>

  **Significance in Backpropagation:**

  **Gradient Calculation:** The derivative of the activation function is used to compute the gradient of the loss function with respect to the output of each neuron in the network. This gradient is then backpropagated through the network to update the weights.

  **Weight Update:** In the backpropagation algorithm, the derivative of the activation function is multiplied with the error signal to compute the gradient of the loss function with respect to the weights. This gradient is used to update the weights of the network, thereby minimizing the error.

  **Non-linearity:** Activation functions introduce non-linearity into the network, allowing it to learn complex patterns in the data. The derivative of the activation function ensures that this non-linearity is preserved during the backpropagation process, enabling the network to learn and adapt effectively.<br/>

## 3. Programming Exercise:<br/>

  **o Implement the Activation Activation Function in Python. Use the following prototype for your function:**

    def Activation_Function_Name(x) :
    # Your implementation

  **Linear Function:**<br/>

    def linear(x):
    return x

  **Sigmoid Function:**

    def sigmoid(x):
    return 1 / (1 + np.exp(-x))

  **Tanh Function:**

    def tanh(x):
    return np.tanh(x)

  **ReLU Function:**

    def relu(x):
    return np.maximum(0, x)

  **Softmax Function:**

    def softmax(x):

    # Subtracting the maximum value for numerical stability
    exp_scores = np.exp(x - np.max(x))  
    return exp_scores / np.sum(exp_scores)

  **Leaky ReLU Function:**

    def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

  **Swish Function:**

    def swish(x):
    return x * sigmoid(x)

  **o Create a small dataset or use an existing one to apply your function and visualize the results.**

  **Generate a dataset for Ativation functions:**

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

 **Using Graph:**

  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/active.PNG)<br/>

  **Using Pie chart:** 

  <br/>![alt text](https://github.com/Rifat-Ahammed/Activation-Function/blob/main/images/pie.PNG)<br/>

  **Installation:**

  To execute this code, you need to install these two Python libraries. 

    # For juoyter notebook, kaggle.
    !pip install numpy
    !pip install matplotlib

  Note For *cmd*, *bash* or *Powershell* just use:

    pip install numpy
    pip install matplotlib

  **Linux package manager:**<br/>

  If you are using the Python version that comes with your Linux distribution, you can install Matplotlib via your package manager, e.g.:

    Debian / Ubuntu: sudo apt-get install python3-matplotlib/numpy

    Fedora: sudo dnf install python3-matplotlib/numpy

    Red Hat: sudo yum install python3-matplotlib/numpy

    Arch: sudo pacman -S python-matplotlib/numpy

## 4. Analysis:

  **o Analyze the advantages and disadvantages of using the Activation Function in neural networks.**<br/>

  Activation functions are the lifeblood of neural networks, introducing non-linearity and enabling learning. However, they present a double-edged sword, offering advantages while introducing potential drawbacks.<br/>

  ## Advantages:

  1. **Non-linearity**: Activation functions introduce non-linearity to the network, enabling it to model complex, non-linear relationships in the data. This allows neural networks to learn and represent highly intricate patterns and make non-trivial decisions.

  2. **Feature Representation**: Activation functions transform input signals into meaningful representations in higher-dimensional space, facilitating the extraction of hierarchical features from raw input data. This helps the network capture relevant information and learn useful abstractions.

  3. **Learning Capability**: By introducing non-linearity, activation functions empower neural networks to learn and model highly intricate relationships between input features and output labels. This makes neural networks suitable for a wide range of tasks, including classification, regression, and sequence modeling.

  4. **Gradient Propagation**: Activation functions play a crucial role in backpropagation, providing gradients that guide the optimization process during training. Properly chosen activation functions ensure efficient gradient flow through the network, leading to faster convergence and better generalization.

  5. **Versatility**: There are various activation functions available, each with its own characteristics and suitability for different types of data and tasks. This versatility allows researchers and practitioners to choose the most appropriate activation function based on the specific requirements of the problem at hand.

  6. **Simplifying Output Interpretation:** Certain activation functions, like sigmoid and tanh, map their outputs to a specific range (e.g., 0-1 for sigmoid, -1 to 1 for tanh). This allows for easier interpretation of the network's predictions in classification tasks. For example, in a binary classification problem, a sigmoid output closer to 1 indicates a higher probability of belonging to one class, while a value closer to 0 suggests the opposite.

  ## Disadvantages:

  1. **Vanishing and Exploding Gradients**: Some activation functions, such as sigmoid and hyperbolic tangent (tanh), are prone to vanishing and exploding gradient problems, especially in deep neural networks. This can hinder the training process by making it difficult for the network to learn long-range dependencies or lead to numerical instability.

  2. **Dead Neurons**: In some cases, neurons may become "dead" or unresponsive if the inputs to the activation function are consistently negative, as in the case of ReLU neurons with negative inputs. This can reduce the capacity of the network to learn and generalize effectively.

  3. **Non-differentiability**: Some activation functions, such as ReLU, are non-differentiable at certain points (e.g., exactly zero for ReLU). While this is not necessarily a problem in practice, it can complicate gradient-based optimization algorithms and require special treatment during training.

  4. **Bias and Variance**: The choice of activation function can introduce bias or variance into the model, affecting its ability to generalize to unseen data. It's important to select activation functions carefully based on the characteristics of the data and the requirements of the task.

  5. **Complexity and Computation**: Certain activation functions, particularly those involving exponential or trigonometric operations (e.g., softmax, tanh), can be computationally expensive, especially when dealing with large datasets or deep architectures. This can increase training time and resource requirements.
  Choosing the right activation function requires careful consideration of these advantages and disadvantages, ensuring your neural network unlocks its full potential.<br/>

  <br/>**o Discuss the impact of the Activation function on gradient descent and the problem of vanishing gradients.**
  