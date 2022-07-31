#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
np.set_printoptions(precision=3)


# In[7]:


class OurNeuralNet(object):
    def __init__(self, X, layers):
        self.X = X
        self.layers = layers
    
    @staticmethod
    def sigmoid(x):
        """
        Argument: value(s) x
        Returns: f(x) where f is sigmoig activativation
        """ 
        # Our activation function: f(x) = 1 / (1 + e^(-x))
        return 1 / (1 + np.exp(-x))

    def parameters_initialization(self):
        """
        Argument:
        - a list of number of neurons in each layer

        Returns:
        params -- python dictionary containing initial parameter values:
        W1 - weight matrix of shape (n1, n0)
        b1 - bias vector of shape (n1, 1)
        W2 - weight matrix of shape (n2, n1)
        b2 - bias vector of shape (n2, 1)
        """    
        
        # Number of neurons in each layer. We just have 3 layers
        n0, n1, n2 = self.layers
        np.random.seed(3)
        
        #Generating parameter values for layer 1
        w1 = np.random.randn(n1,n0) * 0.1
        b1 = np.zeros((n1,1))
        print("w1 shape: ", w1.shape)
        
        #Generating initial parameter values for layer 2
        w2 = np.random.randn(n2,n1) * 0.1
        b2 = np.zeros((n2,1))
        print("w2 shape: ", w2.shape)

        params = {"w1": w1,
                    "b1": b1,
                        "w2": w2,
                          "b2": b2}

        return params
    
    def forward_propagation(self):
        """
        Call parameters_initialization() function
        
        Returns:
        yhat - model output on one forward pass for all the training examples
        layer_ouputs - a dictionary containing model outputs at each layer.
        """
        # Call function to initialize parameters
        parameters = self.parameters_initialization()
        print("X shape: ", self.X.shape)
        w1 = parameters["w1"]
        print("w1 shape: ", w1.shape)
        b1 = parameters["b1"]
        print("b1 shape", b1.shape)
        w2 = parameters["w2"]
        print("w2 shape: ", w2.shape)
        b2 = parameters["b2"]
        print("b2 shape", b2.shape)

        # Perform computations for each layer
        z1 = np.dot(w1, self.X) + b1
        f1 = self.sigmoid(z1)
        print("f1 shape", f1.shape)
        z2 = np.dot(w2, f1) + b2
        print("z2.shape", z2.shape)
        yhat = self.sigmoid(z2)
        print("yhat shape", yhat.shape)

        # Just to make sure that the output is of the dimension
        # we expect
        # It should be a vector of the predictions for the for all examples
        # self.X.shape[1] - number of training examples
        assert(yhat.shape == (1, self.X.shape[1]))
        
        layer_outputs = {"z1": z1,
                 "f1": f1,
                 "z2": z2,
                 "yhat": yhat}

        return yhat, layer_outputs

# Load the data
df = pd.read_csv("https://kipronokoech.github.io/assets/datasets/marks.csv")
#df = pd.read_csv("marks.csv")
X = df.drop(["y"], axis=1) # feature matrix
y = df["y"] # target variable

n0 = X.shape[1] #input size = number of features
n1 = 4 # 4 neurons on the hidden
n2 = 1 # one neuron for output

layers = (n0, n1, n2)
# note: we need X in the dimension X (#features, #training examples)
# therefore we transpose the feature matrix, that is, X.T
s = OurNeuralNet(X= X.T, layers=layers)
y_hat, _ = s.forward_propagation()


# In[ ]:




