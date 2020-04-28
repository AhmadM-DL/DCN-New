import torch
import torch.nn as nn 
import numpy as np
from torch.nn.init import uniform_
from torch.distributions.binomial import  Binomial

class dA(nn.Module):

  def __init__(self, n_visible, n_hidden, corruption_level):
    super(dA, self).__init__()

    self.n_visible = n_visible
    self.n_hidden = n_hidden

    # corruption
    self.corruption_level = corruption_level

    # encoder
    self.encoder = nn.Sequential(
        nn.Linear(n_visible, n_hidden),
        nn.ReLU())    

    #decoder
    self.decoder = nn.Sequential(
        nn.Linear(n_hidden, n_visible),
        nn.ReLU()) 

    self.initialize_weights()

  def initialize_weights(self):
    weights_size = self.encoder[0].weight
    weights_bound = 4 * np.sqrt(6. / (self.n_hidden + self.n_visible))

    tied_weights = nn.Parameter(uniform_(self.encoder[0].weight, a=-weights_bound, b=weights_bound))

    self.encoder[0].weight.data = tied_weights.clone()
    self.decoder[0].weight.data = self.encoder[0].weight.data.transpose(0,1)

    self.encoder[0].bias.data.fill_(0)
    self.decoder[0].bias.data.fill_(0)

  def forward(self,x):

    # add noise
    x = x*Binomial(total_count=1, probs= 1 - self.corruption_level).sample([x.shape[1]])
    x = self.encoder(x)
    x = self.decoder(x)
    
    return x 

class Sdc(torch.nn.Module):
      
    def __init__(self):
      super(Sdc, self).__init__()