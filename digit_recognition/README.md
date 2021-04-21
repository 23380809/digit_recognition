# digit_recognition
digit_recognition using python from scratch without tensorflow. The program is able to reach an accuracy rate of 95%.
## Structure 
### Initialize network
There are 3 fully connected layers with 784, 32, 10 units each layer and each layer uses Python built-in structure to hold the weights.
Weights are set through normal distribution between (0, 1).
[{'weights': [0.7230910569842391, 0.3597793079018069, 0.6794622424433031, 0.18301374299295925]
[‘weights’][0:3] -> weights.   [‘weights][-1] -> bias. 

### **Hidden layer**
Structure has a total of 32 neurons in the first hidden layer. Relu is used as the activation function here. two reasons why I’ve chosen Relu instead of Sigmoid, Relu is definitely more computationally efficient. The second reason is in the most multiple layers cases, Sigmoid function leads to slow convergence, every step it takes will only make a tiny change when performing gradient descent.  In short, Relu converges quicker and computes faster.

### **Output layer**
Ten neurons in output layer used to determine and predict which digit the data was supposed to be.
we take advantage of SoftMax function when predicting one of several outputs. SoftMax function would maximize the normalized values of outputs, which would be the key factor of predicting the right output.

## **Optimizer**

### **Stochastic gradient descent**
problem: could easily stuck at local minima and not getting out of it, need to reload the program again and again to find the exact spot that would converge to global minima. Not so easy to find the perfect learning rate.

### **SGD with momentum**
Not quite satisfied with the convergence rate it gives. The convergence path would usually be heading to a wrong place first.

### **Adagrad**
As the name suggested, adaptive learning rate gradient descent
Better than the above optimizer, but the problem is there’s a slight contradiction to this algorithm. As the error rate is decreasing to the global minima, the algorithm would slower down the learning rate which was intended to let the error getting stuck at global minima. However, the case is that learning rate is being decreased too fast, it would lead to a slow convergence as a result.

### **adadelta**
Revised version of adagrad, By the it removed the effect of adagrad drastically decreasing learning rate and having a decent speed of adjusting learning rate instead. 
