import numpy as np
import math

# This is a two class classification problem which will be solved using the sample neuron
# ----------------------Declaring Variables----------------------
learning_rate = 0.1 # Learning rate
trainig_epochs = 400 # Trainig epochs


# ------------------------Data Preparation-----------------------
# Two features for two classes for trainig
x_class1_feat1 = 1
x_class1_feat2 = 1
x_class2_feat1 = -1
x_class2_feat2 = -1
y_class1 = 0.25 # Target for class 1
y_class2 = -0.25 # Target for class 2


# --------------Defining the Neuron and Activation---------------

def sigmoid(x):
    "Sigmoid function output of a scalar value x"
    return 1.0 / (1.0 + math.exp(-x))

class Neuron(object):
    "This function will take two input x1 and x2 to produce an output y neuron output"
    def __init__(self):

        self.w1 = 0.25 # Initializing the weight1 to 0.25
        self.w2 = 0.25 # Initializing the weight2 to 0.25
        self.b = 0.25 # Initializing the bias to 0.25

    def update_weights_biases (self,grad_w1,grad_w2,grad_b,learning_rate):

        # Gradient update using the gradient descent formula
        self.w1 = self.w1 - learning_rate * grad_w1
        self.w2 = self.w2 - learning_rate * grad_w2
        self.b = self.b - learning_rate * grad_b

        return

    def sigmoid(self,x): # Sigmoid function
        return 1.0 / (1.0 + math.exp(-x))

    def grad_w1(self,x1,output,target): # Gradient of W1
        gradient  =  2 * (output-target) * output*(1-output) * x1 
        return gradient
    
    def grad_w2(self,x2,output,target): # Gradient of W2
        gradient  = 2 * (output-target) * output*(1-output) * x2 
        return gradient
    
    def grad_b(self,output,target): # Gradient of bias
        gradient  = 2 * (output-target) * output*(1-output)
        return gradient


    def forward (self, x1,x2): 
        # forward path
        y_in = np.sum( self.w1*x1 + self.w2*x2 + self.b ) # W*X + b
        y = self.sigmoid(y_in)
        
        return y

def error_function (x,y):
    "Computes the MSE between x and y"
    cost = np.sum((x-y) ** 2) # Mean Squred Error
    return  cost


# Defining the neuron
neuron = Neuron()


# -----------------Traininig Loop for the Neuron-----------------

for i in range(trainig_epochs):

        # Forward class1 and cost
        output1 = neuron.forward(x_class1_feat1,x_class1_feat2 )
        cost1 = error_function (output1,y_class1)

        # Gradient
        grad_w1 = neuron.grad_w1(x_class1_feat1,output1,y_class1)
        grad_w2 = neuron.grad_w2(x_class1_feat2,output1,y_class1)
        grad_b = neuron.grad_b(output1,y_class1)

        # Update
        neuron.update_weights_biases (grad_w1,grad_w2,grad_b,learning_rate)

        # Forward class2 and cost
        output2 = neuron.forward(x_class2_feat1,x_class2_feat2)
        cost2 = error_function (output2,y_class2)
        
        # Gradient
        grad_w1 = neuron.grad_w1(x_class2_feat1,output2,y_class2)
        grad_w2 = neuron.grad_w2(x_class2_feat2,output2,y_class2)
        grad_b = neuron.grad_b(output2,y_class2)

        # Update
        neuron.update_weights_biases (grad_w1,grad_w2,grad_b,learning_rate)
        
        if i == 0:
            print("Initial Iteration values")
            print("Cost =", cost1+cost2)
            print("weight1 = ", neuron.w1)
            print("weight2 = ", neuron.w2)
            print("bias = ", neuron.b)

print("Final Solution")
print("Cost =", cost1+cost2)
print("weight1 = ", neuron.w1)
print("weight2 = ", neuron.w2)
print("bias = ", neuron.b)




