import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4,5"
device = torch.device('cuda'  if torch.cuda.is_available() else 'cpu')
n_class = 10
data_root = "./cifar-10/cifar-10-python"
ARCHITECTURES = ['resnet-110']
