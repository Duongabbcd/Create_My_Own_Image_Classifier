import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json
from collections import OrderedDict
from workspace_utils import active_session

using_gpu = torch.cuda.is_available()

transforms_setting = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

# For using of train.py
def loading_data():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    testval_transforms = transforms_setting

    # Load the datasets with ImageFolder
    trainset = datasets.ImageFolder(train_dir, transform=train_transforms)
    testset = datasets.ImageFolder(test_dir, transform=testval_transforms)
    valset = datasets.ImageFolder(valid_dir, transform=testval_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
    
    return trainloader, testloader, valloader, trainset


# Build and train your network
# Freeze parameters so we don't backprop through them
def build_model(no_output_categories):
    hidden_units = 10240
    model = models.vgg16(pretrained=True)
    
    input_size = 25088
    output_size = 102
    
    for param in model.parameters():
        param.requires_grad = False

    # Defining the fully connected layer that will be trained on the flower images
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(input_size,hidden_units)),
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout(0.5)),
                            ('fc2', nn.Linear(hidden_units,no_output_categories)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    model.classifier = classifier
    
    return model, input_size


# Training the model
def training_model(model, trainloader, valloader, lr, device):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    epochs = 10
    print_every = 20
    running_loss = running_accuracy = 0
    validation_losses, training_losses = [],[]

    # change to cuda
    if device=='gpu':
        model = model.to('cuda')
    else:
        model.to(device)
        
    with active_session():
        for e in range(epochs):
            batches = 0
            model.train()
            for images, labels in trainloader:
                start = time.time()
                batches += 1
                # Moving images & labels to the GPU
                images,labels = images.to(device),labels.to(device)
                # Pushing batch through network, calculating loss & gradient, and updating weights
                log_ps = model.forward(images)
                loss = criterion(log_ps,labels)
                loss.backward()
                optimizer.step()
                # Calculating metrics
                ps = torch.exp(log_ps)
                top_ps, top_class = ps.topk(1,dim=1)
                matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
                accuracy = matches.mean()
                # Resetting optimizer gradient & tracking metrics
                optimizer.zero_grad()
                running_loss += loss.item()
                running_accuracy += accuracy.item()
                # Running the model on the validation set every 5 loops
                if batches%print_every == 0:
                    end = time.time()
                    training_time = end-start
                    start = time.time()
                    # Setting metrics
                    validation_loss = 0
                    validation_accuracy = 0
                    # Turning on evaluation mode & turning off calculation of gradients
                    model.eval()
                    with torch.no_grad():
                        for images,labels in valid_dataloader:
                            images,labels = images.to(device),labels.to(device)
                            log_ps = model.forward(images)
                            loss = criterion(log_ps,labels)
                            ps = torch.exp(log_ps)
                            top_ps, top_class = ps.topk(1,dim=1)
                            matches = (top_class == \
                                   labels.view(*top_class.shape)).type(torch.FloatTensor)
                            accuracy = matches.mean()
                            # Tracking validation metrics
                            validation_loss += loss.item()
                            validation_accuracy += accuracy.item()
                
                    # Tracking metrics
                    end = time.time()
                    validation_time = end-start
                    validation_losses.append(running_loss/print_every)
                    training_losses.append(validation_loss/len(valid_dataloader))
                
                    # Printing Results
                    print(f'Epoch {e+1}/{epochs} | Batch {batches}')
                    print(f'Running Training Loss: {running_loss/print_every:.3f}')
                    print(f'Running Training Accuracy: {running_accuracy/print_every*100:.2f}%')
                    print(f'Validation Loss: {validation_loss/len(valid_dataloader):.3f}')
                    print(f'Validation Accuracy: {validation_accuracy/len(valid_dataloader)*100:.2f}%')
                    print(f'Training Time: {training_time:.3f} seconds for {print_every} batches.')
                    print(f'Validation Time: {validation_time:.3f} seconds.\n')

                    # Resetting metrics & turning on training mode
                    running_loss = running_accuracy = 0
                    model.train()

# Do validation on the test set
def testing_model(model, dataloader):
    test_accuracy = 0
    for images,labels in dataloader:
        model.eval()
        images,labels = images.to(device),labels.to(device)
        log_ps = model.forward(images)
        ps = torch.exp(log_ps)
        top_ps,top_class = ps.topk(1,dim=1)
        matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
        accuracy = matches.mean()
        test_accuracy += accuracy
    print(f'Model Test Accuracy: {test_accuracy/len(dataloader)*100:.2f}%')


# Save the checkpoint 
def save_checkpoints(model, arch, lr, epochs, input_size, hidden_units, class_to_idx, checkpoint_path):
    state = {
            'structure' :arch,
            'learning_rate': lr,
            'epochs': epochs,
            'input_size': input_size,
            'hidden_units':hidden_units,
            'state_dict':model.state_dict(),
            'class_to_idx': class_to_idx
        }
    torch.save(state, checkpoint_path + 'command_checkpoint.pth')
    print('Checkpoint saved in ', checkpoint_path + 'command_checkpoint.pth')

    
# ---------------------------------------------------------------------------------------------------------------------
# For using of predict.py
# Write a function that loads a checkpoint and rebuilds the model
def loading_checkpoint(path):
    
    # Loading the parameters
    state = torch.load(path)
    lr = state['learning_rate']
    input_size = state['input_size']
    structure = state['structure']
    hidden_units = state['hidden_units']
    epochs = state['epochs']
    
    # Building the model from checkpoints
    model,_ = make_model(structure, hidden_units, lr)
    model.class_to_idx = state['class_to_idx']
    model.load_state_dict(state['state_dict'])
    return model


# Inference for classification
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
   
    image_transforms = transforms_setting
    img = image_transforms(pil_image)
    return img

# Class Prediction
def prediction(processed_image, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    model.eval()
    model.cpu()
    processed_image = processed_image.unsqueeze_(0)
    processed_image = processed_image.float()
    
    with torch.no_grad():
        output = model.forward(processed_image)
        probs, classes = torch.topk(input=output, k=topk)
        top_prob = probs.exp()

    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[each] for each in classes.cpu().numpy()[0]]
        
    return top_prob, top_classes
    
# Labeling
def labeling(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name
