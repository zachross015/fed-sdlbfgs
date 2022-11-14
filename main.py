# My stuff
from resnet import resnet18
from cnn import CNN
from sdlbfgs import SdLBFGS
from sdlbfgs_layer import SdLBFGSLayer
<<<<<<< HEAD
from kfac import KFACOptimizer
from shampoo import Shampoo

=======
import time
from collections import OrderedDict
import copy
>>>>>>> af177fba4b46bc531716bedcc27ebf0858d62438
import argparse

# Dataset utilities
import torchvision
import torchvision.transforms as transforms

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim


# Config
train_batch_size = 64
test_batch_size = 64
num_workers = 2
local_epochs = 1
rounds = 1000
local_lr = 0.001
server_lr = 1.0
device='cuda' if torch.cuda.is_available() else "cpu"


# Dataset loading
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


# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--optimizer', required=True)
args = parser.parse_args()

# Definitions
if args.model == 'CNN':
    net = CNN(3, len(classes), 32)
else:
    net = resnet18(len(classes))
net = net.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_sgd = optim.SGD(net.parameters(), lr=local_lr)
if args.optimizer == 'sdlbfgs':
    optimizer = SdLBFGS(net.parameters(), lr=server_lr)
elif args.optimizer == 'sdlbfgs_layer':
    optimizer = SdLBFGSLayer(net.parameters(), lr=server_lr)
elif args.optimizer == 'adam':
    optimizer = optim.Adam(net.parameters())
elif args.optimizer == 'kfac':
    optimizer = KFACOptimizer(net.parameters())
elif args.optimizer == 'shampoo':
    optimizer = Shampoo(net.parameters())
else:
    optimizer = optim.SGD(net.parameters(), lr=0.05)

filename = f'outputs_{net.__class__.__name__}_{optimizer.__class__.__name__}_{rounds}.csv'

def train_test_epoch(epoch, optimizer, local=False):

    running_loss_tr = 0.0
    num_batches_tr = 0
    num_correct_tr = 0.0
    num_samples_tr = 0.0

    start = time.time()

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
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


    net.eval()
    running_loss_te = 0.0
    num_correct_te = 0.0
    num_samples_te = 0.0
    num_batches_te = 0

    with torch.no_grad():
        for x, y in testloader:
            x = x.to(device)
            y = y.to(device)
            scores = net(x)
            running_loss_te += criterion(scores, y)
            _, predictions = scores.max(1)
            num_correct_te += (predictions == y).sum()
            num_samples_te += predictions.size(0)
            num_batches_te += 1

    end = time.time()
    elapsed = end - start
        
    if not local:
        tr_acc = num_correct_tr / num_samples_tr
        tr_loss = running_loss_tr / num_batches_tr
        te_acc = num_correct_te / num_samples_te
        te_loss = running_loss_te / num_batches_te
        with open(filename, 'a+') as f:
            f.write(f'{epoch},{tr_acc:.3f},{tr_loss:.3f},{te_acc:.3f},{te_loss:.3f},{elapsed}\n')


with open(filename, 'a+') as f:
    f.write('epoch,tr_acc,tr_loss,te_acc,te_loss,elapsed\n')
for roun in range(rounds):
    net.train()
    for epoch in range(local_epochs):
        local_grad = train_test_epoch(epoch, optimizer_sgd, local=True)
    train_test_epoch(roun, optimizer)

