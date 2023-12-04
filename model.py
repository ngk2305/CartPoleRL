import torch
import torch.nn as nn
import torch.nn.functional as F
class SimpleModel(nn.Module):
    def __init__(self,action_size,state_size):
        super(SimpleModel, self).__init__()
        self.W1= nn.Linear(state_size,64)
        self.W2 = nn.Linear(64, 16)
        self.W3 = nn.Linear(16, action_size)
        self.relu= nn.ReLU()

    def forward(self,x):
        u1= self.W1(x)
        h1= self.relu(u1)
        u2 = self.W2(h1)
        h2 = self.relu(u2)
        u3 = self.W3(h2)
        #print(u3)
        return u3


