# My stuff
from models import CNN, resnet18
from optims import SdLBFGS, SdLBFGSLayer, KFACOptimizer, Shampoo

import time
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


def get_model(model_name):
    if model_name == 'CNN':
        model = CNN(3, len(classes), 32)
    else:
        model = resnet18(len(classes))
    model = model.to(device)
    return model


def get_optim(optim_name):
    if optim_name == 'sdlbfgs':
        return SdLBFGS
    elif optim_name == 'sdlbfgs_layer':
        return SdLBFGSLayer
    elif optim_name == 'adam':
        return optim.Adam
    elif optim_name == 'kfac':
        return KFACOptimizer
    elif optim_name == 'shampoo':
        return Shampoo
    else:
        return optim.SGD


###########################
# mark: CLIENT DEFINITION #
###########################


class Client:

    def __init__(self, model_name, optim_name, trainset, testset):

        # Init Model
        self.model = get_model(model_name)
        self.optim = get_optim(optim_name)(self.model.parameters(), lr=local_lr)

        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)


    def train_epoch(self):
        running_loss_tr = 0.0
        num_batches_tr = 0
        num_correct_tr = 0.0
        num_samples_tr = 0.0

        for i, data in enumerate(self.trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            self.optim.zero_grad()

            # forward + backward + optimize
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            self.optim.step()

            # Calculate statisics
            running_loss_tr += loss.item()
            num_batches_tr += 1

            with torch.no_grad():
                _, predictions = outputs.max(1)
                num_correct_tr += (predictions == labels).sum()
                num_samples_tr += predictions.size(0)

        return (running_loss_tr / num_batches_tr, num_correct_tr / num_samples_tr)

    def test_epoch(self):
        running_loss_te = 0.0
        num_correct_te = 0.0
        num_samples_te = 0.0
        num_batches_te = 0

        with torch.no_grad():
            for x, y in testloader:
                x = x.to(device)
                y = y.to(device)
                scores = self.model(x)
                running_loss_te += criterion(scores, y)
                _, predictions = scores.max(1)
                num_correct_te += (predictions == y).sum()
                num_samples_te += predictions.size(0)
                num_batches_te += 1

        return (running_loss_te / num_batches_te, num_correct_te / num_samples_te)


clients = []
client_datasets = torch.utils.data.random_split(trainset, [1.0 / num_clients for _ in range(num_clients)], generator=torch.Generator().manual_seed(0))
client_testsets = torch.utils.data.random_split(testset, [1.0 / num_clients for _ in range(num_clients)], generator=torch.Generator().manual_seed(0))

for i in range(num_clients):
    clients.append(Client(args.model, 'SGD', client_datasets[i], client_testsets[i]))


#########################
# mark: LOSS DEFINITION #
#########################


criterion = nn.CrossEntropyLoss()


##############################
# mark: OPTIMIZER DEFINITION #
##############################


# filename = f'tmp_outputs_{server_model.__class__.__name__}_{optimizer.__class__.__name__}_{rounds}.csv'


class Server:

    def __init__(self, model_name, optim_name, clients, testset):

        # Init Model
        self.model = get_model(model_name)
        self.optim = get_optim(optim_name)(self.model.parameters(), lr=server_lr)

        self.testloader = torch.utils.data.DataLoader(
                testset, 
                batch_size=test_batch_size, 
                shuffle=False, 
                num_workers=num_workers)

        self.clients = clients
        self.num_clients = len(clients)

    def train_epoch(self):

        start = time.time()
        params = self.optim.param_groups[0]['params']

        for p in params:
            p.grad = torch.zeros_like(p)
        self.optim.zero_grad()

        # Calculate pseudogradient
        self.model.train()
        for i in range(self.num_clients):

            client = self.clients[i]
            client.model.load_state_dict(copy.deepcopy(self.model.state_dict()))

            for epoch in range(local_epochs):
                client.train_epoch()

            for i, names in enumerate(self.model.named_parameters()):
                (key, _) = names
                self.optim.param_groups[0]['params'][i].grad += self.model.state_dict()[key] - client.model.state_dict()[key]

        for p in params:
            p.grad /= self.num_clients

        # Step using pseudogradient and optimizer
        self.optim.step()

        end = time.time()
        self.elapsed = end - start

    def test_epoch(self):

        self.model.eval()
        running_loss_te = 0.0
        num_correct_te = 0.0
        num_samples_te = 0.0
        num_batches_te = 0

        # Calculate testing loss and accuracy
        with torch.no_grad():
            for x, y in testloader:
                x = x.to(device)
                y = y.to(device)
                scores = self.model(x)
                running_loss_te += criterion(scores, y)
                _, predictions = scores.max(1)
                num_correct_te += (predictions == y).sum()
                num_samples_te += predictions.size(0)
                num_batches_te += 1

        # Print results
        te_acc = num_correct_te / num_samples_te
        te_loss = running_loss_te / num_batches_te

        return (te_loss, te_acc, self.elapsed)

server = Server(args.model, args.optimizer, clients, testset)


# with open(filename, 'a+') as f:
print('epoch,te_acc,te_loss,elapsed\n')

for roun in range(rounds):
    server.train_epoch()
    loss, acc, elapsed = server.test_epoch()
    print(f'{roun},{acc},{loss},{elapsed}')

