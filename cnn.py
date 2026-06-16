import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall

#load datasets
from torchvision import datasets
import torchvision.transforms as transforms

#convert from PIL to torch tensor
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

#feed the data in shuffled batches of 64 images per epoch
dataloader_train = DataLoader(train_data, shuffle = True, batch_size = 64)

#set up nn
class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            #input: 1 * 28 * 28
            nn.Conv2d(1, 32, kernel_size = 3, padding = 1), #learns 32 features by sliding 3x3 box over image
            #32 * 28 * 28
            nn.ELU(), #non linear and doesn't ignore negative gradients
            nn.MaxPool2d(kernel_size = 2), #simplifies Conv2d by taking max from each region (halves data)
            #32 * 14 * 14
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            #64 * 14 * 14
            nn.ELU(),
            nn.MaxPool2d(kernel_size = 2),
            #64 * 7 * 7
            nn.Flatten(), #turns 3D (channel, height, width) data into long 1D vector for linear layer
            #output: 3136 (64*7*7)
        )
        self.classifier = nn.Linear(64*7*7,num_classes)

    #forward pass
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

#set up for training loop
net = Net(num_classes = 10)
criterion = nn.CrossEntropyLoss() #uses softmax to measure error from label
optimizer = optim.Adam(net.parameters(), lr = 0.001)  #updates weights (based on gradients)

#training loop
for epoch in range(7): #low passes to decrease runtime
    for images, labels in dataloader_train:
        optimizer.zero_grad() #reset gradient
        outputs = net(images) #make prediction
        loss = criterion(outputs, labels) #evaluate loss
        loss.backward() #compute gradients
        optimizer.step() #update weights to reduce loss
    print(f"Epoch {epoch + 1}, loss: {loss.item():.4f}")

#feed test data. shuffling has no benefit
dataloader_test = DataLoader(test_data, batch_size=64, shuffle=False)

#evaluation metrics
accuracy_metric = Accuracy(task = "multiclass", num_classes = 10)
precision_metric = Precision(task = "multiclass", num_classes = 10, average = None)
recall_metric = Recall(task = "multiclass", num_classes = 10, average = None)

#evaluation loop
predictions = []
net.eval() #evaluation mode
with torch.no_grad(): #no need to track gradients
    for images, labels in dataloader_test:
        outputs = net(images)
        _, preds = torch.max(outputs, 1) #preds stores index of class with highest score (_ is raw score)
        predictions.extend(preds.tolist()) #flatten batch of predictions into list
        accuracy_metric(preds, labels)
        precision_metric(preds, labels)
        recall_metric(preds, labels)
accuracy = accuracy_metric.compute().item() #tensor -> float
precision = precision_metric.compute().tolist() #tensor -> list
recall = recall_metric.compute().tolist()

#display metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
