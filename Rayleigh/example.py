from keras import backend as K
import numpy as np
import time
import matplotlib.pyplot as plt


print('#'*50)
# kvar = K.variable(value=np.array([[1, 2], [3, 4]]))
# x = np.arange(0,50,1)
# normal = K.random_normal(shape=(1,50), mean=0.0, stddev=1.0, seed=42)
session = K.get_session()
sigma = 1.0
mean = 0.0
x = K.arange(-2, 2, 50)
x.eval(session=session)
print(x)
time.sleep(30)
g1d = K.exp(-(K.pow(x - mean, 2.0) / (2.0 * K.pow(sigma, 2.0)))) * (1.0 
	/ (sigma * K.sqrt(2.0 * 3.1415)))

plt.plot(g1d.eval(session=session))
plt.show()


# a = normal.eval(session=session)
# print(a)
# print(x)
# plt.plot(x,a,linewidth=2, color='r')
# plt.show()
# print(K.shape(kvar).eval(session=session))

