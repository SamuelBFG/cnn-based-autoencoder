from keras import backend as K
import numpy as np
import time

sess = K.get_session()

zeros = K.zeros(shape=(1,50,2))
K.eval(zeros)
print(zeros)
# time.sleep(20)
for _ in np.arange(0, 2):
	print('X')
	x = K.square(K.random_normal(shape=(1,50,2), mean=0.0, stddev=np.sqrt(1 / 2), seed=42).eval(session=sess))
	print(x.eval(session=sess))
	zeros = zeros + x
	print('#'*50)

print('#'*50)
print('ZEROS')
print(zeros.eval(session=sess))
print(K.shape(zeros).eval(session=sess))
h = K.sqrt(zeros).eval(session=sess)
print(h)
print(h.shape)

print('fezes')