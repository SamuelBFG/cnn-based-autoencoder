	from keras import backend as K
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.special import gamma

sess = K.get_session()

zeros = K.zeros(shape=(1,100,2))
K.eval(zeros)
print(zeros)
# time.sleep(20)
alpha = 7/4
mu = 2
for _ in np.arange(0, mu):
	print('X')
	x = K.square(K.random_normal(shape=(1,100,2), mean=0.0, stddev=np.sqrt(1 / 2*(mu)), seed=42)).eval(session=sess)
	print(x)
	zeros = zeros + x
	print('#'*50)

print('#'*50)
print('ZEROS')
print(zeros.eval(session=sess))
print(K.shape(zeros).eval(session=sess))
h = K.pow(zeros, alpha).eval(session=sess)

print(h[0][0][0])

print('#'*50)
print('#'*50)
print('#'*50)

print(h[0][:])
print(h.shape)
print('fezes')

_, bins, _ = plt.hist(h, 100, density=True)
A_1 = alpha
A_2 = mu**mu
A_3 = bins**(alpha*mu-1)
A = A_1 * A_2 * A_3
B_1 = gamma(mu)
B_2 = np.exp(mu*bins**alpha)
B = B_1 * B_2
pdfAlphaMu = A/B

# pdfAlphaMu = A*B*C
plt.plot(bins, pdfAlphaMu, linewidth=2, color='r',
    label=r'$\Omega_{\alpha-\mu} = $' + '{}'.format(1))
plt.title(r'$\alpha-\mu$ pdf')
plt.show()