import torch
x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

#forward path and compute the loss
y_hat = w * x
loss = (y_hat - y)**2   #y_hat-y mu 2

print(loss)
loss.backward()
print(w.grad)

##update weights
#do next forward and backward


