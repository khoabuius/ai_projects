import torch
import numpy as np
import torch.nn as nn
from sklearn import datasets
import matplotlib.pyplot as plt

# prepare data
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
print(f'x_numpy: {x_numpy}, x_numpy shape: {x_numpy.shape}') #For your x_numpy, since you used n_features=1, it will be (100, 1) indicating 100 rows and 1 column.
print(f'y_numpy: {y_numpy}, y_numpy shape: {y_numpy.shape}') #For y_numpy, it will typically be (100,), a 1D array with 100 elements, or sometimes (100, 1) depending on how it's generated.
x = torch.from_numpy(x_numpy).float()
y = torch.from_numpy(y_numpy).float()
y = y.view(y.shape[0], 1)
print(f'y: {y}') # Convert y to tensor 100 rows 1 column (100, 1)

x_sample, x_features = x.shape
print(f'x_sample: {x_sample}, x_features: {x_features}')
# model
input_size = x_features
output_size = 1
model = nn.Linear(input_size, output_size)
# loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# training loop
num_epochs = 100
for epoch in range(num_epochs):
    #forward pass
    y_pred = model(x)
    loss = criterion(y_pred, y)
    # backward pass
    loss.backward()

    # optimizer
    optimizer.step()

    optimizer.zero_grad()
    w, b = model.parameters()
    # print(f'Epoch {epoch}, weight: {w[0][0].item():.8f} ,Loss: {loss.item():.4f}')

# Plot
#The red circles ('ro') represent your original data points — the actual data you've collected or generated.
#The blue line ('b') shows the model's predicted output for those data points — how well the model fits or approximates the data.
predicted = model(x).detach().numpy()
plt.plot(x_numpy, y_numpy, 'ro')
plt.plot(x_numpy, predicted, 'b')
plt.show()
'''
Common marker style codes:
Code
Description
'.'
point marker
'o'
circle marker
's'
square marker
'D'
diamond marker
'x'
x marker
'+'
plus marker

Common color codes:

Code
Color
'r' red

'b' blue

'g' green


'k' black


'y' yellow


'm' magenta


'c' cyan


Example:          
plt.plot(x, y, 'gs')  # green squares
plt.plot(x, y, 'b+')  # blue plus signs
'''
