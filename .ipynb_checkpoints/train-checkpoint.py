import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from PIL import Image
import argparse
import processing


# Argparser Arguments
parser = argparse.ArgumentParser()
parser.add_argument('data_directory', action='store',default = 'flowers', help='Set directory to load training data, e.g., "flowers"')
parser.add_argument('--save_dir', action='store', dest='cp_path', default='checkpoints/', help='path of checkpoint')

parser.add_argument('--arch', action='store', dest='arch', default='vgg16', choices={"vgg16", "densenet161"}, help='architecture of the network')
parser.add_argument('--hidden_units', action='store', nargs=2, default=[10240, 1024], dest='hidden_units', type=int,
                    help='Enter 2 hidden units of the network, input -> --hidden_units 256 256 | output-> [512, 256]')
parser.add_argument('--epochs', action='store', dest='epochs', default=3, type=int, help='(int) number of epochs while training')

parser.add_argument('--category_names', action="store", dest="category_names", default='cat_to_name.json')
parser.add_argument('--learning_rate', action='store', nargs='?', default=0.001, type=float, dest='learning_rate', help='(float) learning rate of the network')
parser.add_argument('--gpu', action='store_true', default=False, dest='boolean_t', help='Set a switch to use GPU')

results = parser.parse_args()

checkpoint_path = results.cp_path
arch = results.arch
hidden_units = results.hidden_units
epochs = results.epochs
lr = results.learning_rate
gpu = results.boolean_t
print_every = 30

category_names = results.category_names
cat_to_name = processing.labeling(category_names)
no_output_categories = len(cat_to_name)

if gpu==True:
    using_gpu = torch.cuda.is_available()
    device = 'gpu'
    print('GPU On');
else:
    print('GPU Off');
    device = 'cpu'
    
    
# Loading Dataset
trainloader, testloader, valloader, trainset  = processing.loading_data()
class_to_idx = trainset.class_to_idx

# Network Setup
model, input_size = processing.build_model(no_output_categories)

# Training Model
processing.training_model(model, trainloader, valloader, lr, device)

# Testing Model
processing.testing_model(model, testloader)

# Saving Checkpoint
processing.save_checkpoints(model, arch, lr, epochs, input_size, hidden_units, checkpoint_path)