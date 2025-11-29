import torch
#neural network module
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

x = torch.empty(1,2,2)
print(x.shape)

# Operation with tensors2
x = torch.ones(2,2)
y = torch.ones(2,2)

z = x + y

print(x)
print(y)
print(z)

# Subtraction
z = x - y
z = torch.sub(x,y)

print("\n", x)
print(y)
print(z)

# Multiplication
z = x * y
print("\n multiplication with operator", z)
z = torch.mul(x,y)
print('\n Multiplication with pytorch function :-' ,z)

# Division
z = x / y
z = torch.div(x,y)
print('\n Division :-' , z)

#Autograd
x = torch.randn(3, requires_grad=True)
y = x + 2

print("X :-", x)
print(y)
print(y.grad_fn)


# Linear regression f(x) = w * x + b  , (b = 0 in this case)

x = torch.tensor([1,2,3,4,5,6,7,8],dtype=torch.float32)
y = torch.tensor([2,4,6,8,10,12,14,16],dtype=torch.float32)

#Keeping track of this tensor
w = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

#Model Output
def forward(x):
    return w * x

# loss = MSE
def loss(y,y_pred):
    return ((y_pred - y) ** 2).mean()

x_test = 5.0

#The result will be zero because w (weights) is 0
print(f'\n\nPrediction before training : f({x_test}) = {forward(x_test).item():.3f}')

# Training
learning_rate = 0.01
n_epochs = 100

for epochs in range(n_epochs):
    # predict = forward pass
    y_pred = forward(x)

    # Loss
    l = loss(y,y_pred)

    #calculate gradients = backward pass
    l.backward()

    #update weights
    with torch.no_grad():
        w -= learning_rate * w.grad

    # zero the gradients after updating
    w.grad.zero_()

    if (epochs + 1) % 10 == 0:
        print(f'Epoch {epochs + 1} , w = {w.item():.3f} , loss = {l.item():.3f}')

print(f'Prediction after training: f({x_test}) = {forward(x_test).item():.3f}')



# Model , Loss and optimizer
x = torch.tensor([[1],[2],[3],[4],[5],[6],[7],[8]],dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8],[10],[12],[14],[16]],dtype=torch.float32)

n_samples,n_features = x.shape
print(f'n_samples : {n_samples} , n_features : {n_features}')

x_test = torch.tensor([5],dtype=torch.float32)

class LinearRegression(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegression,self).__init__()
        self.lin = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.lin(x)

input_size , output_size = n_features , n_features

model = LinearRegression(input_size,output_size)

print(f'Prediction before the training : f({x_test}) = {forward(x_test).item():.3f}')

# Define Loss and optimizer
learning_rate = 0.01
n_epochs = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

for epochs in range(n_epochs):
    #predict = forward pass with our model
    y_predicted = model(x)

    # loss
    l = loss(y,y_predicted)

    #calculate gradients = backward pass
    l.backward()

    # update weights
    optimizer.step()

    # zero the gradients after updating
    optimizer.zero_grad()

    if (epochs + 1) % 10 == 0:
        print(f'Epoch : {epochs + 1} , loss = {l.item()} , w = {w.item()}')

print(f'Prediction after training :- f({x_test}) = {model(x_test).item():.3f}')


#         ----------------------- Neural Network ---------------------------

# Hyper-paramters or nerual network layers
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset  = torchvision.datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),
                                            download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',train=False,
                                          transform=transforms.ToTensor()) # Transform to convert into tensors

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


# Fully connected neural network with one hidden layer
class NeuralNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNetwork,self).__init__()
        self.ll = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        out = self.ll(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax in the end
        return out

model = NeuralNetwork(input_size,hidden_size,num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

#Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images = images.reshape(-1,28*28)

        #Forward pass and loss calculation
        outputs = model(images)
        loss = criterion(outputs,labels)

        #Backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 10 == 0:
            print(f'Epoch : {epoch + 1} / {num_epochs} , Step : {i + 1} / {n_total_steps} , loss : {loss.item():.4f}')

# Testing the model
with torch.no_grad():
    n_correct = 0
    n_samples = len(test_loader.dataset)

    for images,labels in test_loader:
        images = images.reshape(-1,28*28)

        outputs = model(images)

        #max returns
        _,predicted = torch.max(outputs,1)
        n_correct += (predicted == labels).sum().item()

    acc = n_correct / n_samples
    print(f'Accurate of the network on the {n_samples} test images : {100 * acc}')


# ---------------------- Convolutional neutral network ------------------------

# Hyper-Parameters
num_epochs = 10
batch_size = 32
learning_rate = 0.001

transform = transforms.Compose([
    transforms.ToTensor() ,
    transforms.Normalize((0.5,0.5,0.5) , (0.5,0.5,0.5))
])

# CIFAR 10 Dataset

train_dataset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=32,shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=32,shuffle=False)


class ConvNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,3)
        self.conv3 = nn.Conv2d(64,64,3)
        self.fc1 = nn.Linear(64*4*4,64)
        self.fc2 = nn.Linear(64,10)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = ConvNeuralNetwork()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    running_loss = 0.0

    for i,(images,labels) in enumerate(train_loader):
        #Forward Pass
        outputs = model(images)
        loss = criterion(outputs,labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f'[{epoch + 1}] loss : {running_loss / n_total_steps:.3f}')

print('Finished Training')

#Evaluating CNN

with torch.no_grad():
    n_correct = 0
    n_samples = len(test_loader.dataset)

    for images,labels in test_loader:
        outputs = model(images)

        _,predicted = torch.max(outputs,1)
        n_correct += (predicted == labels).sum().item()

    acc = 100 * n_correct / n_samples
    print(f'Accuracy of the model :- {acc}')



