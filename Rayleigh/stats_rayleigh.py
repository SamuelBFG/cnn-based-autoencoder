import numpy as np
import matplotlib.pyplot as plt
import time

# GAUSSIAN
# mu, sigma = 0, .5
# N = 10**6
# print(N)
# x = np.random.normal(mu, sigma, N)
# count, bins, ignored = plt.hist(x, 30, density=True)
# print(bins)
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
# 	               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
# 	               linewidth=2, color='r')
# plt.title('Gaussian pdf')
# plt.show()


# RAYLEIGH
# mu, sigma = 0, .5
# N = 10**6
# x = np.random.normal(mu, sigma, N)
# y = np.random.normal(mu, sigma, N)
# r = np.sqrt(x**2 + y**2)
# print(r.shape)
# count, bins, ignored = plt.hist(r, 100, density=True)
# pdfRayleigh = bins/(sigma**2) * np.exp( - bins**2 / (2 * sigma**2))
# plt.plot(bins, pdfRayleigh, linewidth=2, color='r')
# plt.title('Rayleigh pdf')
# plt.show()

# ALPHA-MU
sigma = .5
N = 10**6
alpha = 2
mu = 3


rAlphaMu = []
for i in np.arange(0, mu):
	print (i)
	x = np.random.normal(0, sigma, N)
	y = np.random.normal(0, sigma, N)
	r = np.sqrt(x**2 + y**2)
	rAlphaMu.append(r)
print(np.shape(rAlphaMu))

count, bins, ignored = plt.hist(rAlphaMu, 30, density=True)
pdfAlphaMu = alpha * mu**mu * bins**(alpha*mu -1) / ()
plt.plot(bins)
plt.title('Rayleigh pdf')
plt.show()