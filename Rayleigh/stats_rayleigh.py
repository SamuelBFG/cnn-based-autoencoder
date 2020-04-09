import os
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import gamma

# GAUSSIAN
mean, var = 0, 1
N = 10**6
sigmaGauss = np.sqrt(var)
print('#'*50)
print('Gaussian Standard Deviation = {}'.format(np.sqrt(var)))
print('Gaussian Variance = {}'.format(sigmaGauss**2))
x = np.random.normal(mean, sigmaGauss, N)
_, bins, _ = plt.hist(x, 100, density=True)
plt.plot(bins, 1/(sigmaGauss * np.sqrt(2 * np.pi)) *
	               np.exp( - (bins - mean)**2 / (2 * sigmaGauss**2) ),
	               linewidth=2, color='r', label=r'$\sigma^2 = {}$.'.format(sigmaGauss))

plt.title(r'Gaussian pdf ($\sigma_G = {}$)'.format(sigmaGauss))
plt.text(0, 0.5, '$\sigma_G^2 = {}$'.format(sigmaGauss**2))
plt.legend(loc='upper right')
plt.show()


# RAYLEIGH
OmegaRayleigh = 2 * sigmaGauss**2
print('#'*50)
print('OmegaRayleigh = {}'.format(OmegaRayleigh))
xRayleigh = np.random.normal(mean, np.sqrt(OmegaRayleigh/2), N)
yRayleigh = np.random.normal(mean, np.sqrt(OmegaRayleigh/2), N)
rRayleigh = np.sqrt(xRayleigh**2 + yRayleigh**2)
count, bins, ignored = plt.hist(rRayleigh, 100, density=True)

pdfRayleigh = (2*bins/OmegaRayleigh) * np.exp(-bins**2/OmegaRayleigh)
plt.plot(bins, pdfRayleigh, linewidth=2, color='r', 
	label=r'$\Omega_{ray} = 2\sigma_G^2 = $' + '{}.'.format(OmegaRayleigh))

# pdfRayleighNorm = bins/(sigmaGauss**2) * np.exp(-bins**2/(2 * sigmaGauss**2))
# plt.plot(bins, pdfRayleighNorm, linewidth=2, color='g', 
# 	label=r'$ \sigma = {}$ (scale parameter).'.format(sigmaGauss))

plt.legend(loc='upper right')
plt.title(r'Rayleigh pdf)')
plt.show()



# GAMMA
'''
Can be seen as a sum of N squared Rayleigh RVs ~ Rayleigh(sigmaGauss) or Rayleigh(OmegaRayleigh/2)
'''
k = 5 # k = N Rayleigh RVs (SHAPE PARAMETER)
theta = OmegaRayleigh # (SCALE PARAMETER)

rGamma = []
for _ in np.arange(0, k):
	xGamma = np.random.normal(mean, np.sqrt(OmegaRayleigh/2), N)
	yGamma = np.random.normal(mean, np.sqrt(OmegaRayleigh/2), N)
	rGamma.append(xGamma**2 + yGamma**2)
rGamma = sum(rGamma)

count, bins, ignored = plt.hist(rGamma, 100, density=True)

A = bins**(k-1) * np.exp(-bins/theta)
B = gamma(k) * theta**k
pdfGamma = A/B
plt.plot(bins, pdfGamma, linewidth=2, color='r', 
	label=r'$\Omega_{ray} = 2\sigma_G^2 = $' + '{}.'.format(OmegaRayleigh))


plt.legend(loc='upper right')

plt.title(r'Gamma pdf ($\sigma_G^2 = {}$)'.format(sigmaGauss))

plt.show()


# # NAKAGAMI-m
mu = k
# OmegaNakagami = OmegaRayleigh*mu

print('m = {}'.format(mu))
print('OmegaNakagami = {}'.format(OmegaNakagami))
rNakagami = []
for _ in np.arange(0, mu):
	x = np.random.normal(0, np.sqrt(OmegaRayleigh/2), N)
	y = np.random.normal(0, np.sqrt(OmegaRayleigh/2), N)
	rNakagami.append(x**2 + y**2)

rNakagami = np.sqrt(sum(rNakagami))
rNakagami = np.sqrt(rGamma)
count, bins, ignored = plt.hist(rNakagami, 100, density=True)

A = (mu / OmegaNakagami)**mu
B = 2*bins**(2*mu-1) / gamma(mu)
C = np.exp(-mu*bins**2/OmegaNakagami**2)

pdfNakagami = A*B*C

plt.plot(bins, pdfNakagami, linewidth=2, color='r')
plt.title(r'Nakagami-m pdf ($\sigma_G^2 = {}$)'.format(sigmaGauss))
plt.show()



# # ALPHA-MU
# sigma = .5
# N = 10**6
# alpha = 2
# mu = 3


# rAlphaMu = []
# for i in np.arange(0, mu):
# 	print (i)
# 	x = np.random.normal(0, sigma, N)
# 	y = np.random.normal(0, sigma, N)
# 	r = np.sqrt(x**2 + y**2)
# 	rAlphaMu.append(r)
# print(np.shape(rAlphaMu))

# count, bins, ignored = plt.hist(rAlphaMu, 30, density=True)
# pdfAlphaMu = alpha * mu**mu * bins**(alpha*mu -1) / ()
# plt.plot(bins)
# plt.title('Rayleigh pdf')
# plt.show()