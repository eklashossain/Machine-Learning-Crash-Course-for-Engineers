# ----------------------------Modules----------------------------
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
from numpy.random import randn
from matplotlib import pyplot as plt
import math
import torch.nn.functional as F


# ---------------------------Variables---------------------------
batch_s = 256 # batchsize
Iterations = 5000 # iteraiton for training
plot_epoch = 4999 # final output printing epoch
learning_rate = 0.001 # learning Rate
d_step = 10 # generator training steps
g_step = 10 # discriminator training steps


# -------------------Dataset Using an Equation-------------------
# We want to fit a path to the following equation:
# 7x^2 + 2x + 1

def functions(x):
    return  7*x*x + 2*x + 1

def data_generation():

    'This function generates the data of batch size = batch_s'

    data = []
    x = 20 * randn(batch_s) # random inputs

    for i in range(batch_s):
        y = functions(x[i])
        data.append([x[i], y]) # dataset

    return torch.FloatTensor(data)

def plotting(real, fake, epoch):

    'plotting the real and fake data' 

    x, y = zip(*fake.tolist())
    plt.scatter(x, y, label='Generated Data')
    x, y = zip(*real.tolist())
    plt.scatter(x, y, label='Original Data')
    plt.legend(loc='upper left')
    plt.xlabel("inputs")
    plt.savefig('GAN.png', bbox_inches='tight')
    plt.show()


# -------------Defining Generator and Discriminator--------------

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        "Generator Model 3-layer fully connected"
        self.layer1 = nn.Linear(4, 20)
        self.layer2 = nn.Linear(20, 10)
        self.layer3 = nn.Linear(10, 2)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

generator = Generator()


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        "Dicriminator Model 3-layer fully connected"
        self.layer1 = nn.Linear(2, 20)
        self.drop1  =  nn.Dropout(0.4)
     
        self.layer2 = nn.Linear(20, 10)
        self.drop2  =  nn.Dropout(0.4)
     
        self.layer3 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.leaky_relu(self.drop1(self.layer1(x)))
        x = F.leaky_relu(self.drop2(self.layer2(x)))
        x = torch.sigmoid(self.layer3(x))
        return x

discriminator = Discriminator()


# Setting up the models
generator = Generator()
discriminator = Discriminator()



# Define Optimizer 
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)


# -----------------------Training Function-----------------------

def train(Iterations,optimizer_D,optimizer_G,discrimator,generator):

    "this function trains both generator and discriminator"
    # Set the models to training mode
    discriminator.train()
    generator.train()
    loss_f = nn.BCELoss() # BCE loss function

    for epoch in range(Iterations):
        # discriminator training
        for d_steps in range(d_step):

            data = data_generation()
            z =  torch.randn([batch_s, 4])
           
        
            # no gradient computation at this stage
            fake_data = generator(z).detach()
    
        
            sizes = data.size()[0]
            optimizer_D.zero_grad()
    
            #      the real data update
            prediction_real = discriminator(data)
            d_loss = loss_f(prediction_real, torch.ones(sizes, 1).fill_(0.95))
            d_loss.backward()
    
            #     fake data update
            prediction_generated = discriminator(fake_data)
            loss_generated = loss_f(prediction_generated, torch.zeros(sizes, 1))
            loss_generated.backward()

            optimizer_D.step()
    
        for g_steps in range(g_step):
            z =  torch.randn([batch_s, 4])

        
            fake_data = generator(z)
         
            optimizer_G.zero_grad()
    
            #     Run the generated data through the discriminator
            prediction = discriminator(fake_data)

            #     Train the generator with the smooth target, i.e. the target is 0.95
            loss_gen = loss_f(prediction, torch.ones(sizes, 1).fill_(0.95))
            loss_gen.backward()

            optimizer_G.step()

        print("epoch:", epoch)
        print("Genrator Loss:",d_loss)
        print("Discriminator Loss:",loss_gen)

        if((epoch+1) % plot_epoch == 0):        
            plotting(data, fake_data, epoch)
         

# ----------------------------Training---------------------------
train(Iterations,optimizer_D,optimizer_G,discriminator,generator)