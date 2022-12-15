import torch.nn as nn
import torch
import time
import copy


class Server:

    def __init__(
                 self,
                 model,
                 optim,
                 clients,
                 testloader,
                 criterion=nn.CrossEntropyLoss(),
                 local_epochs=1
                 ):

        # Init Model
        self.model = model
        self.optim = optim
        self.clients = clients
        self.num_clients = len(clients)
        self.testloader = testloader
        self.criterion = criterion
        self.local_epochs = local_epochs

    def update_buffers(self):
        pass

    def zero_grad(self):
        params = self.optim.param_groups[0]['params']
        for p in params:
            p.grad = torch.zeros_like(p)
        self.optim.zero_grad()

    def update_params(self):
        params = self.optim.param_groups[0]['params']
        # Update Pseudogradients
        for i, (name, _) in enumerate(self.model.named_parameters()):
            for j in range(self.num_clients):
                params[i].grad += self.model.state_dict()[name] - self.clients[j].model.state_dict()[name]
            params[i].grad /= self.num_clients

    def backwards(self):

        self.update_params()
        self.update_buffers()

    def move_server_state_to_clients(self):
        for i in range(self.num_clients):
            client = self.clients[i]
            client.model.load_state_dict(copy.deepcopy(self.model.state_dict()))

    def train_clients(self):
        self.move_server_state_to_clients()

        # Calculate pseudogradient
        self.model.train()
        for i in range(self.num_clients):
            for epoch in range(self.local_epochs):
                self.clients[i].train_epoch()

    def train_epoch(self):

        start = time.time()

        self.zero_grad()
        self.train_clients()

        self.backwards()

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


class ResNetServer(Server):

    def update_buffers(self):

        with torch.no_grad():

            buffers = {}
            state_dict = self.model.state_dict()

            # Zero out all the buffers (e.g. running_mean, running_var)
            for name, _ in self.model.named_buffers():

                # Zero the current buffer
                state_dict[name] = torch.zeros_like(self.model.state_dict()[name])

                # Construct the buffer
                buffers[name] = []
                for i in range(self.num_clients):
                    client = self.clients[i]
                    buffers[name].append(client.model.state_dict()[name])
                buffers[name] = torch.stack(buffers[name])

                if 'num_batches_tracked' in name:
                    state_dict[name] = buffers[name].mean(dim=0, dtype=torch.float).type(torch.long)
                else:
                    state_dict[name] = buffers[name].mean(dim=0)

            self.model.load_state_dict(state_dict)

