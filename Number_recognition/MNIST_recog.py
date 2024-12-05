import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# Defining Parameters
input_size = 784
hidden_size =  256
num_classes = 10
num_epochs = 4
batch_size = 50
learning_rate = 0.001

# MNIST dataset
training_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor())

train_loader = DataLoader(training_dataset, batch_size= batch_size, shuffle= True)
test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle=False)

for i, (sample, label) in enumerate(train_loader):
    if i == 6:
        break
    plt.subplot(2,3,i+1)
    plt.imshow(sample[i][0], cmap='gray')
plt.savefig('./example_data.png')
plt.close()

# NN architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_layer_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)

# defining Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
losses = []
# Training loop:
for epoch in range(num_epochs):
    for i, (sample, label) in enumerate(train_loader):
        sample = sample.reshape(-1, 28*28).to(device)
        label = label.to(device)

        outputs= model(sample)
        loss = criterion(outputs, label)
        losses.append(loss.item())
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%50 == 0:
            print(f'epoch {epoch+1} step {i+1}/{len(train_loader)} loss {loss.item():.4f}')

#plot the losses
plt.plot(range(len(losses)), losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss vs Iterations')
plt.savefig('Loss.png')
plt.close()

# Test loop
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for (sample, label) in test_loader:
        sample = sample.reshape(-1, 28*28).to(device)
        label = label.to(device)
        predicted_outputs = model(sample)
        _, predicted = torch.max(predicted_outputs.detach(), 1)

        n_samples += label.size(0)
        n_correct += (predicted == label).sum().item()
    acc = 100 * (n_correct) / (n_samples)

    print(f' accuracy of the model is: {acc}')













   



