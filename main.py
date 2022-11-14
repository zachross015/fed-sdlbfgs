# My stuff
from resnet import resnet18
from cnn import CNN
from sdlbfgs import SdLBFGS
from sdlbfgs_layer import SdLBFGSLayer
from kfac import KFACOptimizer
from shampoo import Shampoo

import time
from collections import OrderedDict
import copy
import argparse
import random
import numpy as np

# Dataset utilities
import torchvision
import torchvision.transforms as transforms

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim


#######################
# mark: CONFIGURATION #
#######################


train_batch_size = 64
test_batch_size = 64
num_workers = 2

local_epochs = 1
local_lr = 0.001
num_clients = 1

rounds = 500
server_lr = 1.0

device = 'cuda' if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


#########################
# mark: DATASET LOADING #
#########################


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            normalize
            ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
        ]))
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


##############################
# mark: COMMAND LINE PARSING #
##############################


parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--optimizer', required=True)
args = parser.parse_args()


##########################
# mark: MODEL DEFINITION #
##########################


if args.model == 'CNN':
    server_model = CNN(3, len(classes), 32)
else:
    server_model = resnet18(len(classes))
server_model = server_model.to(device)

client_models = []
for _ in range(num_clients):
    if args.model == 'CNN':
        client_model = CNN(3, len(classes), 32)
    else:
        client_model = resnet18(len(classes))
    client_model.to(device)
    client_models.append(client_model)

client_optims = []
for i in range(num_clients):
    client_optims.append(optim.SGD(client_models[i].parameters(), lr=local_lr))


#########################
# mark: LOSS DEFINITION #
#########################


criterion = nn.CrossEntropyLoss()


##############################
# mark: OPTIMIZER DEFINITION #
##############################


if args.optimizer == 'sdlbfgs':
    optimizer = SdLBFGS(server_model.parameters(), lr=server_lr)
elif args.optimizer == 'sdlbfgs_layer':
    optimizer = SdLBFGSLayer(server_model.parameters(), lr=server_lr)
elif args.optimizer == 'adam':
    optimizer = optim.Adam(server_model.parameters())
elif args.optimizer == 'kfac':
    optimizer = KFACOptimizer(server_model, lr=0.001)
    optimizer.acc_stats = True
elif args.optimizer == 'shampoo':
    optimizer = Shampoo(server_model.parameters())
else:
    optimizer = optim.SGD(server_model.parameters(), lr=0.05)


filename = f'tmp_outputs_{server_model.__class__.__name__}_{optimizer.__class__.__name__}_{rounds}.csv'


def train_test_epoch_local(model, optimizer):

    running_loss_tr = 0.0
    num_batches_tr = 0
    num_correct_tr = 0.0
    num_samples_tr = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = server_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        # print statistics
        running_loss_tr += loss.item()
        num_batches_tr += 1

        with torch.no_grad():
            _, predictions = outputs.max(1)
            num_correct_tr += (predictions == labels).sum()
            num_samples_tr += predictions.size(0)

    server_model.eval()
    running_loss_te = 0.0
    num_correct_te = 0.0
    num_samples_te = 0.0
    num_batches_te = 0

    with torch.no_grad():
        for x, y in testloader:
            x = x.to(device)
            y = y.to(device)
            scores = server_model(x)
            running_loss_te += criterion(scores, y)
            _, predictions = scores.max(1)
            num_correct_te += (predictions == y).sum()
            num_samples_te += predictions.size(0)
            num_batches_te += 1


def train_test_epoch_global(epoch, optimizer):

    start = time.time()
    params = optimizer.param_groups[0]['params']

    for p in params:
        p.grad = torch.zeros_like(p)
    optimizer.zero_grad()

    # Calculate pseudogradient
    server_model.train()
    for i in range(num_clients):
        model = client_models[i]
        optim = client_optims[i]
        model.load_state_dict(copy.deepcopy(server_model.state_dict()))
        for epoch in range(local_epochs):
            train_test_epoch_local(model, optim)

        for i, names in enumerate(server_model.named_parameters()):
            (key, _) = names
            optimizer.param_groups[0]['params'][i].grad += server_model.state_dict()[key] - model.state_dict()[key]

    for p in params:
        p.grad /= num_clients


    # Step using pseudogradient and optimizer
    optimizer.step()

    server_model.eval()
    running_loss_te = 0.0
    num_correct_te = 0.0
    num_samples_te = 0.0
    num_batches_te = 0

    # Calculate testing loss and accuracy
    with torch.no_grad():
        for x, y in testloader:
            x = x.to(device)
            y = y.to(device)
            scores = server_model(x)
            running_loss_te += criterion(scores, y)
            _, predictions = scores.max(1)
            num_correct_te += (predictions == y).sum()
            num_samples_te += predictions.size(0)
            num_batches_te += 1

    end = time.time()
    elapsed = end - start

    # Print results
    te_acc = num_correct_te / num_samples_te
    te_loss = running_loss_te / num_batches_te
    # with open(filename, 'a+') as f:
    print(f'{epoch},{te_acc:.3f},{te_loss:.3f},{elapsed}\n')


# with open(filename, 'a+') as f:
print('epoch,te_acc,te_loss,elapsed\n')


for roun in range(rounds):
    train_test_epoch_global(roun, optimizer)
