#!/usr/bin/env python3

# Imports
import numpy as np 
      
# Each row is a training example, each column is a feature  [X1, X2, X3]
X=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
X1=np.array(([0,1,1],[0,1,1],[0,0,1],[1,0,1]), dtype=float)
y=np.array(([0],[1],[1],[0]), dtype=float)
print(X.shape[1])
# Define useful functions    

# Activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)

# Class definition
class NeuralNetwork:
    def __init__(self, x,y):
        self.input = x
        self.weights1= np.random.rand(self.input.shape[1],4) # considering we have 4 nodes in the hidden layer
        self.weights2= np.random.rand(4,5) # considering we have 4 nodes in the hidden layer
        self.weights3 = np.random.rand(5,1)
        self.y = y
        self.output = np. zeros(y.shape)
        
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.layer3 = sigmoid(np.dot(self.layer2, self.weights3))
        return self.layer3
        
    def backprop(self):
        d_error3=2*(self.y -self.output)
        d_weights3 = np.dot(self.layer2.T,d_error3*sigmoid_derivative(self.output))
        #print(d_weights3.shape)
        d_error2=np.dot(d_error3, self.weights3.T)
        d_weights2 = np.dot(self.layer1.T, d_error2*sigmoid_derivative(self.layer2))
        #print(d_weights2.shape)
        d_error1=np.dot(d_error2, self.weights2.T)
        d_weights1 = np.dot(self.input.T, d_error1*sigmoid_derivative(self.layer1))
        #print(d_weights1.shape)
    
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.weights3 += d_weights3

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()
        
    def test(self, X1) :
        self.input = X1
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.layer3 = sigmoid(np.dot(self.layer2, self.weights3))
        return self.layer3
        

NN = NeuralNetwork(X,y)
for i in range(1500): # trains the NN 1,000 times
    if i % 100 ==0: 
        print ("for iteration # " + str(i) + "\n")
        print ("Input : \n" + str(X))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(NN.feedforward()))
        print ("Loss: \n" + str(np.mean(np.square(y - NN.feedforward())))) # mean sum squared loss
        print ("\n")
  
    NN.train(X, y)
print ("Predicted Output: \n" + str(NN.test(X1)))
