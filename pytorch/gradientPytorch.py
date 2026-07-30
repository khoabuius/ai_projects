import torch
from numpy.conftest import dtype

x = torch.rand(3, requires_grad=True)
print(x)
y = x + 2
print(y)
z = y*y*2
print(z)
# z = z.mean() #average elements
# print(z)
v = torch.tensor([1.0, 0.1, 0.01], dtype = torch.float16)
z.backward(v) #dz/dx
print(x.grad)

print('--------------')


# 1. Setup inputs and a target
x = torch.tensor([2.0], requires_grad=True)
target = torch.tensor([10.0])

# 2. Forward pass: Model prediction
prediction = x * 3  # Prediction is 6.0

# 3. Calculate loss
loss = (prediction - target) ** 2  # (6.0 - 10.0)^2 = 16.0
print(f"Current Loss: {loss.item()}")  # Outputs: 16.0

# 4. Backward pass: Calculate slopes
loss.backward()

# 5. Inspect the direction of change
print(f"Gradient (Direction): {x.grad.item()}")  # Outputs: -24.0

'''
How to Interpret the Direction (.grad)The gradient value tells you how the loss behaves 
if you increase the value of your parameter:
- Negative Gradient (-24.0): An inverse relationship. Increasing \(x\) will decrease the loss. Therefore, your optimizer knows it needs to add to \(x\) to minimize error.
- Positive Gradient: A direct relationship. Increasing the parameter will increase the loss. Your optimizer knows it needs to subtract from the parameter.
- Zero Gradient: The parameter is at an optimal peak or valley. Changing it will not change the loss.
'''

print('--------------')

weight =torch.ones(3, requires_grad=True)

for item in range(3):
    model_out_put = (weight*3).sum()
    model_out_put.backward()
    print(weight.grad)
    weight.grad.zero_()


print('--------------')
# Data: input features and target values
x = torch.tensor([1.0, 2.0, 3.0])  # input features
target = torch.tensor([2.0, 4.0, 6.0])  # true targets

# Initialize parameters (weights and bias)
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# Forward pass: compute predictions
y_pred = w * x + b

# Compute loss (Mean Squared Error)
loss = ((y_pred - target) ** 2).mean()

print("Loss:", loss.item())



# Compute gradients
loss.backward()

# Access gradients
print("Gradient of w:", w.grad.item())
print("Gradient of b:", b.grad.item())