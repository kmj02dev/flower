"""$project_name: A Flower Baseline."""

import torch
import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """Do forward."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def create_model(**kwargs):
    return Net()

def train(net, trainloader, device, local_epochs, lr, momentum, **kwargs):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    net.train()
    running_loss = 0.0
    for _ in range(local_epochs):
        for batch in trainloader:
            X, y = batch['X'], batch['y']
            optimizer.zero_grad()
            loss = criterion(net(X.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device, **kwargs):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            X, y = batch['X'].to(device), batch['y'].to(device)
            outputs = net(X)
            loss += criterion(outputs, y).item()
            correct += (torch.max(outputs.data, 1)[1] == y).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
