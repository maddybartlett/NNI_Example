import numpy as np
import torch
from copy import deepcopy
import matplotlib.pyplot as plt

class RobustNetwork(torch.nn.Module):
    def __init__(self, nodes=[2, 100, 1]):
        super().__init__()
        self.lyrs = torch.nn.ModuleList()
        for k,n in enumerate(nodes[:-2]):
            self.lyrs.append(torch.nn.Linear(n, nodes[k+1]))
            self.lyrs.append(torch.nn.ReLU())
        self.lyrs.append(torch.nn.Linear(nodes[-2], nodes[-1]))
        self.lyrs.append(torch.nn.Sigmoid())
        self.loss_fcn = torch.nn.BCELoss(reduction='mean')
        self.losses = []
    
    def forward(self, x):
        y = x
        for lyr in self.lyrs:
            y = lyr(y)
        return y

    # A helper function can be used to compute the Frob norm
    def weight_penalty(self):
        penalty = 0.
        for p in self.parameters():
            penalty += torch.sum(p**2)
        return penalty
    
    def learn(self, x, t, epochs=100, lr=0.1, weight_decay=0.):
        for epoch in range(epochs):
            y = self(x)
            
            # Add Frobenius norm of weights to the loss
            loss = self.loss_fcn(y.squeeze(), t.squeeze()) + 0.5*weight_decay*self.weight_penalty()                
            
            self.losses.append(loss.item())
            self.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in self.parameters():
                    p -= lr*p.grad
