import torch
import torch.nn as nn

class Client:

    def __init__(self, model, optim, trainloader, testloader, criterion=nn.CrossEntropyLoss()):

        # Init Model
        self.model = model
        self.optim = optim 
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion


    def train_epoch(self):
        running_loss_tr = 0.0
        num_batches_tr = 0
        num_correct_tr = 0.0
        num_samples_tr = 0.0

        for i, data in enumerate(self.trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(self.model.device), labels.to(self.model.device)

            # zero the parameter gradients
            self.optim.zero_grad()

            # forward + backward + optimize
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()

            self.optim.step()

            # Calculate statisics
            running_loss_tr += loss.item()
            num_batches_tr += 1

            with torch.no_grad():
                _, predictions = outputs.max(1)
                num_correct_tr += (predictions == labels).sum()
                num_samples_tr += predictions.size(0)

        # __import__('pdb').set_trace()

        return (running_loss_tr / num_batches_tr, num_correct_tr / num_samples_tr)

    def test_epoch(self):
        running_loss_te = 0.0
        num_correct_te = 0.0
        num_samples_te = 0.0
        num_batches_te = 0

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

        return (running_loss_te / num_batches_te, num_correct_te / num_samples_te)

