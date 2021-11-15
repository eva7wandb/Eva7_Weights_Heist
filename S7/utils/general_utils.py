import torch

def setup_env():
  SEED = 1
  cuda = torch.cuda.is_available()
  
  torch.manual_seed(SEED)
  if cuda:
      torch.cuda.manual_seed(SEED)
  
  device = torch.device("cuda" if cuda else "cpu")
  return cuda, device

