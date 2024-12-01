import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns


data = datasets.load_iris()

x = data.data
y = data.target

y[y != 0] = -1
y[y == 0] = 1

X_df = pd.DataFrame(x, columns=data.feature_names)
y_df = pd.Series(y, name='Label')
df = pd.concat([X_df, y_df], axis=1)
corr_matrix = df.corr()
plt.figure(figsize=(16, 16))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix between Features and Labels")
plt.savefig("corr_matrix.png")
plt.close()

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

n_samples, n_features = X_train.shape
training_dataset = TensorDataset(X_train, y_train)
batch_size = 10
train_loader = DataLoader(training_dataset, batch_size = batch_size, shuffle = True)

model = nn.Linear(n_features, 1)

def loss(x,y):
    output = model(x)
    y = y.view(-1,1)
    l = 1 - y*output
    net_loss = torch.max(torch.tensor(0),l)
    return net_loss.mean()

learning_rate = 0.01
n_iters = 250

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
losses = []

for epoch in range(n_iters):
    for i, (inputs, labels) in enumerate(train_loader):
        net_loss = loss(inputs, labels)
        losses.append(net_loss.item())
        net_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i+1)%2 == 0:
            print(f'epoch {epoch+1}, step {i+1}, loss {net_loss}')

plt.plot(range(len(losses)), losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss vs Iterations')
plt.savefig('Loss.png')
plt.close()

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size= batch_size)

accuracy = 0
for i, (inputs, labels) in enumerate(test_loader):
    output = model(inputs)
    predicted = torch.sign(output).squeeze()
    accuracy += (predicted == labels).sum().item()

n_test = X_test.shape[0]


print('#'*10)
for name, param in model.named_parameters():
    print(f"{name}: {param}")
print(f'accuracy {(accuracy/n_test)*100}%')







