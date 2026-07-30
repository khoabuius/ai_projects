import numpy as np

# f = w * x
# f = 2 * x

X = np.array([1, 2, 3, 4], dtype=np.float32)
y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0

# Model prediction
def  forward(x):
    return w * x

# loss = mean error linear
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

# gradient
# mean error linear = 1/N * (w*x - y)**2
# dj/dw = 1/N 2x (w*x - y_pred)
def gradient(x, y, y_pred):
    return np.dot(2*x, y_pred -y).mean()

print(f'Predict before train f(5)= {forward(5):.3f}')

# training
learning_rate = 0.01
n_inter = 20

for i in range(n_inter):
    # Prediction with forward pass
    y_pred = forward(X)

    # Loss
    l = loss(y, y_pred)

    #gradient
    dw = gradient(X, y, y_pred)

    #update weight
    w -= learning_rate * dw
    if i%1 == 0:
        print(f'Iteration {i}: weight= {w:.8f}, loss ={l:.8f}')

print(f'Predict after train f(5)= {forward(5):.3f}')