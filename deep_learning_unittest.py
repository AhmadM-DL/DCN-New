"""
Created on Tuesday April 20 2020
@author: Ahmad Mustapha (amm90@mail.aub.edu)

I created this model to serve as a modular and re-usable Deep Learning
Test Suit it contains methods that tests different aspects of Deep Learning
modules. For example one method is to test whether the parameter changed after
an optimization/training step or not.
"""

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset

class RandomVisionDataset(VisionDataset):
    def __init__(self, input_size, data_length, n_classes, transform=None):
      self.len = data_length
      self.data = torch.randn(data_length, *input_size)
      self.targets = torch.empty(data_length, dtype=torch.long).random_(n_classes)
      self.imgs =  [ ("dummy/path/class/img%d.png"%i, self.targets[i]) for i in range(self.len)]  
      self.transform = transform   
      device = torch.device("cpu")

    def __getitem__(self, index):
      """
      Args:
          index (int): Index

      Returns:
          tuple: (sample, target) where target is class_index of the target class.
      """
      path, target = self.imgs[index]
      sample = self.data[index]
      if self.transform is not None:
          sample = self.transform(sample)
      return sample, target

    def __len__(self):
      return len(self.imgs)

class RandomDataset(Dataset):

    def __init__(self, input_size, data_length, n_classes):
      self.len = data_length
      self.data = torch.randn(data_length, *input_size)
      self.targets = torch.empty(data_length, dtype=torch.long).random_(n_classes)        
      device = torch.device("cpu")

    def __getitem__(self, index):
      return (self.data[index], self.targets[index])

    def __len__(self):
      return self.len

def do_train_step(model, loss_fn, optim, batch, device):
    """Run a training step on model for a given batch of data
    Parameters of the model accumulate gradients and the optimizer performs
    a gradient update on the parameters
    Parameters
    ----------
    model : torch.nn.Module
      torch model, an instance of torch.nn.Module
    loss_fn : function
      a loss function from torch.nn.functional
    optim : torch.optim.Optimizer
      an optimizer instance
    batch : list
      a 2 element list of inputs and labels, to be fed to the model
    """

    # put model in train mode
    model.train()
    model.to(device)

    #  run one forward + backward step
    # clear gradient
    optim.zero_grad()
    # inputs and targets
    inputs, targets = batch[0], batch[1]
    # move data to DEVICE
    inputs = inputs.to(device)
    targets = targets.to(device)
    # forward
    likelihood = model(inputs)
    # calc loss
    loss = loss_fn(likelihood, targets)
    # backward
    loss.backward()
    # optimization step
    optim.step()


def do_forward_step(model, batch, device):
    """Run a forward step of model for a given batch of data
    Parameters
    ----------
    model : torch.nn.Module
      torch model, an instance of torch.nn.Module
    batch : list
      a 2 element list of inputs and labels, to be fed to the model
    Returns
    -------
    torch.tensor
      output of model's forward function
    """

    # put model in eval mode
    model.eval()
    model.to(device)

    with torch.no_grad():
        # inputs and targets
        inputs = batch[0]
        # move data to DEVICE
        inputs = inputs.to(device)
        # forward
        return model(inputs)


def test_param_change(vars_change, model, loss_fn, optim, batch, device, params=None):
    """Check if given variables (params) change or not during training
    If parameters (params) aren't provided, check all parameters.
    Parameters
    ----------
    vars_change : bool
      a flag which controls the check for change or not change
    model : torch.nn.Module
      torch model, an instance of torch.nn.Module
    loss_fn : function
      a loss function from torch.nn.functional
    optim : torch.optim.Optimizer
      an optimizer instance
    batch : list
      a 2 element list of inputs and labels, to be fed to the model
    params : list, optional
      list of parameters of form (name, variable)
    """

    if params is None:
        # get a list of params that are allowed to change
        params = [np for np in model.named_parameters() if np[1].requires_grad]

    # take a copy
    initial_params = [(name, p.clone()) for (name, p) in params]

    # run a training step
    do_train_step(model, loss_fn, optim, batch, device)

    # check if variables have changed
    for (_, p0), (name, p1) in zip(initial_params, params):
        if vars_change:
            assert not torch.equal(p0.to(device), p1.to(device))
        else:
            assert torch.equal(p0.to(device), p1.to(device))

def test_param_tied(param1, param2, model, loss_fn, optim, batch, device):
    """Check if given variables (params) are tied together, i.e. they 
    change during training but they are always equal during training.

    Parameters
    ----------
    param1 : torch.nn.tensor
      the first parameter
    param2: torch.nn.tensor
      the second parameter
    model : torch.nn.Module
      torch model, an instance of torch.nn.Module
    loss_fn : function
      a loss function from torch.nn.functional
    optim : torch.optim.Optimizer
      an optimizer instance
    batch : list
      a 2 element list of inputs and labels, to be fed to the model
    """

    # take a copy
    params = (param1, param2)
    initial_params = (param1.clone(), param2.clone())

    # run a training step
    do_train_step(model, loss_fn, optim, batch, device)

    # check that paramters have changed after training
    assert not torch.equal( params[0].to(device), initial_params[0].to(device))
    assert not torch.equal( params[1].to(device), initial_params[1].to(device))

    # check that parameters are equal before and after training
    assert torch.equal(params[0], params[1])
    assert torch.equal(initial_params[0], initial_params[1])

