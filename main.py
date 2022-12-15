# My stuff
from common import get_model, get_optim
from federated import Client, Server, ResNetServer

import argparse
import random
import numpy as np

# Dataset utilities
import torchvision
import torchvision.transforms as transforms

# PyTorch
import torch
import torch.nn as nn


##############################
# mark: COMMAND LINE PARSING #
##############################


parser = argparse.ArgumentParser()

parser.add_argument('--model', required=True)

# Global config
parser.add_argument('--train_batch_size', default=64, type=int)
parser.add_argument('--test_batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=2, type=int)

# Local Config
parser.add_argument('--local_epochs', default=1, type=int)
parser.add_argument('--local_lr', default=0.001, type=float)
parser.add_argument('--num_clients', default=1, type=int)

# Server config
parser.add_argument('--optimizer', required=True)
parser.add_argument('--rounds', default=500, type=int)
parser.add_argument('--server_lr', default=1.0, type=float)
parser.add_argument('--seed', default=0, type=int)

args = parser.parse_args()


#######################
# mark: CONFIGURATION #
#######################


train_batch_size = args.train_batch_size
test_batch_size = args.test_batch_size
num_workers = args.num_workers

local_epochs = args.local_epochs
local_lr = args.local_lr
num_clients = args.num_clients

rounds = args.rounds
server_lr = args.server_lr

device = 'cuda' if torch.cuda.is_available() else "cpu"

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

criterion = nn.CrossEntropyLoss()


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


###########################
# mark: CLIENT DEFINITION #
###########################


clients = []
client_datasets = torch.utils.data.random_split(trainset, [1.0 / num_clients for _ in range(num_clients)], generator=torch.Generator().manual_seed(0))
client_testsets = torch.utils.data.random_split(testset, [1.0 / num_clients for _ in range(num_clients)], generator=torch.Generator().manual_seed(0))

for i in range(num_clients):

    # Get model and store device information in model
    model = get_model(args.model)
    model = model.to(device)
    model.device = device

    optim = get_optim('SGD', model, lr=local_lr)

    trainloader = torch.utils.data.DataLoader(
            client_datasets[i],
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers)

    testloader = torch.utils.data.DataLoader(
            client_testsets[i],
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers)

    clients.append(Client(model, optim, trainloader, testloader))


###########################
# mark: SERVER DEFINITION #
###########################


model = get_model(args.model)
model = model.to(device)
model.device = device

optim = get_optim(args.optimizer, model, lr=server_lr)
if args.model == 'CNN':
    server_type = Server
else:
    server_type = ResNetServer
server = server_type(
        model,
        optim,
        clients,
        testloader,
        local_epochs=local_epochs,
        criterion=criterion)


# with open(filename, 'a+') as f:
print('epoch,te_acc,te_loss,elapsed')

for roun in range(rounds):
    server.train_epoch()
    loss, acc, elapsed = server.test_epoch()
    print(f'{roun},{acc},{loss},{elapsed}')
