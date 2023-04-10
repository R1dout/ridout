import torch.nn as nn

def select_loss_fuc(args):
  type2loss = {
    'l1':nn.L1Loss(),
    'ce':nn.CrossEntropyLoss(),
    'mse':nn.MSELoss()
  }
  loss_fuc = type2loss[args.loss_fuc]
  return loss_fuc
