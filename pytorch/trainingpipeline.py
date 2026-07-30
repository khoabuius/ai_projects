import torch
import torch.nn as nn

# Design our model (input, output size, forward pass
# Construct loss and optimizer
# Training loop
   #- forward pass: compute prediction
   #- backward pass: compute gradient
   #- Update our weights

# f = w * x
# f = 2 * x

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(f'n_samples={n_samples}, n_features={n_features}')
input_size = n_features
output_size = n_features

class LinearCustom(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearCustom, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.linear(x)

model = LinearCustom(input_size, output_size)
print(f'Predict before train f(5)= {model(X_test).item():.3f}')

# training
learning_rate = 0.01
n_inter = 2000
loss = nn.MSELoss() # mean square error
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for i in range(n_inter):
    # Prediction with forward pass
    y_pred = model(X)

    # Loss
    l = loss(y, y_pred)
    # gradient backward pass
    l.backward() # dl/dw
    optimizer.step()
    #zero gradient so no accumulate it
    optimizer.zero_grad()

    if i%1 == 0:
        w,b = model.parameters()
        print(f'Iteration {i}: weight= {w[0][0].item():.8f}, loss ={l:.8f}')

print(f'Predict after train f(5)= {model(X_test).item():.3f}')