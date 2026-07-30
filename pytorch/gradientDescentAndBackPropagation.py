import torch

# f = w * x
# f = 2 * x

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w =  torch.tensor(0.0, requires_grad=True)

# Model prediction
def  forward(x):
    return w * x

# loss = mean error linear
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

print(f'Predict before train f(5)= {forward(5):.3f}')

# training
learning_rate = 0.01
n_inter = 100

for i in range(n_inter):
    # Prediction with forward pass
    y_pred = forward(X)

    # Loss
    l = loss(y, y_pred)
    # gradient backward pass
    l.backward() # dl/dw
    with torch.no_grad():
        #update weight
        w -= learning_rate * w.grad
    #zero gradient so no accumulate it
    w.grad.zero_()

    if i%1 == 0:
        print(f'Iteration {i}: weight= {w:.8f}, loss ={l:.8f}')

print(f'Predict after train f(5)= {forward(5):.3f}')