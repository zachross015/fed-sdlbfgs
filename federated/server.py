import torch.nn as nn
import torch
import time
import copy

class Server:

    def __init__(self, model, optim, clients, testloader, criterion=nn.CrossEntropyLoss(), local_epochs=1):

        # Init Model
        self.model = model
        self.optim = optim
        self.clients = clients
        self.num_clients = len(clients)
        self.testloader = testloader
        self.criterion = criterion
        self.local_epochs=local_epochs

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

            for epoch in range(self.local_epochs):
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
            for x, y in self.testloader:
                x = x.to(self.model.device)
                y = y.to(self.model.device)
                scores = self.model(x)
                running_loss_te += self.criterion(scores, y)
                _, predictions = scores.max(1)
                num_correct_te += (predictions == y).sum()
                num_samples_te += predictions.size(0)
                num_batches_te += 1

        # Print results
        te_acc = num_correct_te / num_samples_te
        te_loss = running_loss_te / num_batches_te

        return (te_loss, te_acc, self.elapsed)
