"""
import numpy as np
x = np.asarray([np.empty([3], dtype = int),np.empty([3], dtype = int)])
print(x.shape)
x = x[...]
print(x.shape)
x=x.T
print(x.shape)
val_y=np.empty([3], dtype = int)
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x, val_y)
print(reg.coef_)
"""
import torch
import numpy as np
data = np.array([0.9, 0.3, 0.7])
tensor = torch.FloatTensor(data.T)

target = np.array([1,0,1])
target = torch.FloatTensor(data.T)

out=torch.nn.functional.binary_cross_entropy_with_logits(tensor,target,reduce=False)
print(out)