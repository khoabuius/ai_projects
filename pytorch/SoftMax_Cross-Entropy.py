import torch
import torch.nn as nn
import numpy as np


x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print(f'outputs = {outputs}')

# Cross - Entropy
'''
One-hot encoding is a method used to change text or category labels into numbers 
(binary lists of 0s and 1s) so that computer and machine learning models can understand them. 
It creates a separate column for each choice, 
putting a 1 where that choice is true and a 0 everywhere else.
How It WorksImagine you have a color list with three choices: Red, Green, and Blue.
Instead of writing words, the system makes three new columns: Is_Red, Is_Green, and Is_Blue.
If your item is Red, it becomes [1, 0, 0] (1 for red, 0 for the others).
If your item is Blue, it becomes [0, 0, 1] (1 for blue, 0 for the others)
'''

loss = nn.CrossEntropyLoss()
y = torch.tensor([0])

# y prediction = num_samples * num_classes = 1 * 3
y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
y_pred_bad = torch.tensor([[0.1, 2.0, 0.3]])

l1 = loss(y_pred_good, y)
l2 = loss(y_pred_bad, y)
print(f'l1 = {l1.item()}')
print(f'l2 = {l2.item()}')
'''
dim=1 tells torch.max to find the maximum value across columns (i.e., along axis 1).

For a 2D tensor with shape [num_samples, num_classes]:

dim=0 → max across rows (compares samples for each class)
dim=1 → max across columns (compares classes for each sample)
In your case with [y_pred_good = [[2.0, 1.0, 0.1]]](http://vscodecontentref/4) (shape [1, 3]):
This matches the expected format for nn.CrossEntropyLoss: [num_samples, num_classes].

Syntax	Shape	Dimensions
[2.0, 1.0, 0.1]	[3]	1D — just 3 values, no batch dimension
[[2.0, 1.0, 0.1]]	[1, 3]	2D — 1 sample with 3 class scores
[[2.0, 1.0, 0.1], [0.1, 2.0, 0.3]]	[2, 3]	2D — 2 samples with 3 class scores

dim=1 → scan across [2.0, 1.0, 0.1] → max is 2.0 at index 0
torch.max with dim returns two values:

_ (the max value itself, e.g. 2.0)
pred_good (the index of the max, e.g. 0 — which is the predicted class)
'''
_,pred_good = torch.max(y_pred_good, dim=1)
_,pred_bad = torch.max(y_pred_bad, dim=1)
print(f'pred_good = {pred_good.item()}')
print(f'pred_bad = {pred_bad.item()}')

######## for 3 samples
y = torch.tensor([2, 0, 1])
# y prediction = num_samples * num_classes = 3 * 3
y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]])

l1 = loss(y_pred_good, y)
l2 = loss(y_pred_bad, y)
print(f'l1 = {l1.item()}')
print(f'l2 = {l2.item()}')

_,pred_good = torch.max(y_pred_good, dim=1)
_,pred_bad = torch.max(y_pred_bad, dim=1)

'''
torch.max(tensor, dim=1) returns the index of the max value in each row:

y_pred_good = [[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]]

Sample	Scores	Max value	Index of max
Row 0	[0.1, 1.0, 2.1]	2.1	2
Row 1	[2.0, 1.0, 0.1]	2.0	0
Row 2	[0.1, 3.0, 0.1]	3.0	1
→ pred_good = tensor([2, 0, 1]) ✅ matches y = [2, 0, 1] perfectly

y_pred_bad = [[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]]

Sample	Scores	Max value	Index of max
Row 0	[2.1, 1.0, 0.1]	2.1	0
Row 1	[0.1, 1.0, 2.1]	2.1	2
Row 2	[0.1, 3.0, 0.1]	3.0	1
→ pred_bad = tensor([0, 2, 1]) ❌ only sample 2 matches y = [2, 0, 1]

The index represents the predicted class — whichever class got the highest raw score (logit).
'''
print(f'pred_good = {pred_good}')
print(f'pred_bad = {pred_bad}')


############# Multi classes problem to predict cat or dog?

class MulNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MulNeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        #No softmax
        return out

model = MulNeuralNet(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss() # applies Softmax

############# Binary classes [0 or 1] problem to predict is it a dog?

class BinaryNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryNeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        #use sigmoid at the end
        return torch.sigmoid(out)

model = BinaryNeuralNet(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss() # applies Softmax
