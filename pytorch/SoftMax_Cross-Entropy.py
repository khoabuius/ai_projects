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
