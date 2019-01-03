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