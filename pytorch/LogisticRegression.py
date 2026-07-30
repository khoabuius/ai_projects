import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Prepare data
bc = datasets.load_breast_cancer() #Load and return the breast cancer Wisconsin dataset (classification).
x, y = bc.data, bc.target # Loads the Wisconsin Breast Cancer dataset (569 samples, 30 features), x = feature matrix (569 × 30), y = labels (0=malignant, 1=benign)
n_samples, n_features = x.shape
print(f'n_samples: {n_samples}, n_features: {n_features}')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234) # 80% training (455 samples), 20% testing (114 samples), random_state=1234 ensures reproducibility
print(f' x_train: {x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}')
# scale
sc = StandardScaler()
# Normalizes each feature to zero mean and unit variance, Critical for gradient descent convergence — without it, features with large ranges dominate
x_train = sc.fit_transform(x_train) # learn mean/std from train, then scale
x_test = sc.transform(x_test) # scale test using train's mean/std

x_train = torch.from_numpy(x_train).float()
x_test = torch.from_numpy(x_test).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()

#Converts NumPy arrays → PyTorch tensors (.float() = float32)
#.view(..., 1) reshapes y from a 1D vector to a column vector (required for BCELoss)
y_train = y_train.view(y_train.shape[0], 1) #Returns a new tensor with the same data as the :attr:`self`, 1 row return
y_test = y_test.view(y_test.shape[0], 1) # reshape from [455] to [455, 1]
print(f' y_train tensor: {y_train.shape}, y_test tensor: {y_test.shape}')
# Setup model
# f = w*x + b,
class LogisticRegression(nn.Module):
    def __init__(self, input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_features, 1) # 30 inputs → 1 output
    def forward(self, x):
        '''

        :param x:
        :return:
        '''
        y_pred = torch.sigmoid(self.linear(x)) #Sigmoid is a function that squashes any real number into the range (0, 1)
        return y_pred

model = LogisticRegression(n_features)
# Loss and optimizer

criterion = nn.BCELoss() # Binary Cross-Entropy Loss
# SGD with learning rate 0.01 — updates weights by w -= lr * gradient
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent
# Training loop
num_epochs = 100


'''
Each epoch:

Feed all training data through the model
Measure how wrong the predictions are
Calculate gradient of loss w.r.t. each weight
Adjust weights in the direction that reduces loss
Reset gradients (PyTorch accumulates them by default)
'''
for epoch in range(num_epochs):
    #forward pass and calculation
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    #backward
    loss.backward()
    #update stage
    optimizer.step()
    #empty gradient
    optimizer.zero_grad()

    if epoch % 10 == 0:
      print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

with torch.no_grad():
    y_pred = model(x_test)

    y_pred_cls = y_pred.round() #converts probabilities (e.g., 0.87) to class labels (1)
    print(f'y_pred: {y_pred.shape}, y_test: {y_test.shape}')
    '''
    Breaking it down piece by piece:

Part	    What it does	   Example
y_pred_cls	Predicted classes   (0 or 1)	[1, 0, 1, 1, 0]
.eq(y_test)	Element-wise comparison: True if prediction == actual	[T, T, F, T, T]
.sum()	Counts how many True (correct predictions)	4
.item()	Converts single-element tensor to a Python number	4 (int)
/ y_test.shape[0]	Divides by total number of test samples	4 / 5 = 0.8
    '''
    acc = y_pred_cls.eq(y_test).sum().item() / y_test.shape[0] #element-wise comparison, counts correct predictions
    print(f'Test Accuracy: {acc:.4f}')

