from keras import backend as K
import numpy as np
import time
import matplotlib.pyplot as plt


print('#'*50)
# kvar = K.variable(value=np.array([[1, 2], [3, 4]]))
# x = np.arange(0,50,1)
# normal = K.random_normal(shape=(1,50), mean=0.0, stddev=1.0, seed=42)
sess = K.get_session()
sigma = 1.0
mean = 0.0
x = K.arange(start=-5.0, stop=5.0, step=000.1).eval(session=sess)
print(x)
print(x.shape)
print('#'*50)
# time.sleep(30)
# g1d = K.exp(-(K.pow(x - mean, 2.0) / (2.0 * K.pow(sigma, 2.0)))) * (1.0 
# 	/ (sigma * K.sqrt(2.0 * 3.1415)))
# normal = K.random_normal(shape=(K.shape(x)), mean=0.0, stddev=1.0, seed=42)
# normal2 = K.random_normal_variable(shape=K.shape(x), mean=0.0, scale=1.0, seed=42)
normal = K.random_normal(shape=K.shape(x), mean=0.0, stddev=1/2, seed=42).eval(session=sess)
normal2 = K.random_normal(shape=K.shape(x), mean=0.0, stddev=1/2, seed=41).eval(session=sess)
# print('## 1a Gaussian Shape:')
# print(K.shape(normal))
# print('## 1a Gaussian Values:')
# print(normal)

r = K.pow(normal, 2).eval(session=sess) + K.pow(normal2, 2).eval(session=sess)
print('## R Shape:')
print(K.shape(r))
print('## R Values:')
print(r)
# time.sleep(100)
plt.yscale('log')
plt.plot(x, r, '--b')

# plt.plot(x, normal2, '-r')
plt.show()


# a = normal.eval(session=session)
# print(a)
# print(x)
# plt.plot(x,a,linewidth=2, color='r')
# plt.show()
# print(K.shape(kvar).eval(session=session))

