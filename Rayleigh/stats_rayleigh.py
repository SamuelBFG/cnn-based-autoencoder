import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import gamma

# GAUSSIAN
mean, sigmaGauss = 0, .5
N = 10**6

x = np.random.normal(mean, sigmaGauss, N)
count, bins, ignored = plt.hist(x, 30, density=True)
print(bins)
plt.plot(bins, 1/(sigmaGauss * np.sqrt(2 * np.pi)) *
	               np.exp( - (bins - mean)**2 / (2 * sigmaGauss**2) ),
	               linewidth=2, color='r')
plt.title('Gaussian pdf')
plt.show()


# RAYLEIGH
OmegaRayleigh = 2*sigmaGauss
xRayleigh = np.random.normal(mean, OmegaRayleigh, N)
yRayleigh = np.random.normal(mean, OmegaRayleigh, N)
rRayleigh = np.sqrt(xRayleigh**2 + yRayleigh**2)
count, bins, ignored = plt.hist(rRayleigh, 100, density=True)
pdfRayleigh = bins/(OmegaRayleigh**2) * np.exp( - bins**2 / (2 * OmegaRayleigh**2))
plt.plot(bins, pdfRayleigh, linewidth=2, color='r')
plt.title('Rayleigh pdf')
plt.show()

# NAKAGAMI-m
mu = 5

OmegaNakagami = 1

OmegaRayleigh = OmegaNakagami/mu

rNakagami = []
for _ in np.arange(0, mu):
	print('#'*50)
	x = np.random.normal(0, OmegaRayleigh/2, N)
	y = np.random.normal(0, OmegaRayleigh/2, N)
	# x = np.random.normal(0, OmegaNakagami/(2*mu), N)
	# y = np.random.normal(0, OmegaNakagami/(2*mu), N)
	r = np.sqrt(x**2 + y**2)
	rNakagami.append(r)
rNakagami = sum(rNakagami)
count, bins, ignored = plt.hist(rNakagami, 100, density=True)
# pdfNakagami = 2*mu**mu/gamma(mu)*(bins**(2*mu-1.0))*np.exp(-mu*bins*bins)
# pdfNakagami = 2 * mu**mu / gamma(mu) * bins**(2*mu-1) * np.exp(-mu*bins**2)
A = (mu / OmegaNakagami)**mu
B = 2*bins**(2*mu-1) / gamma(mu)
C = np.exp(-mu*bins**2/OmegaNakagami**2)
pdfNakagami = A*B*C
# pdfNakagami = (mu**mu * OmegaNakagami**(-mu)) * (2*bins**(2*mu-1)) / (gamma(mu)) * np.exp((-mu * bins**2) / OmegaNakagami**2)
# r = nakagami.rvs(.5, size=1000)
plt.plot(bins, pdfNakagami, linewidth=2, color='r')
plt.title('Nakagami-m pdf')
plt.show()

# # plt.plot(bins, pdfRayleigh)


# plt.show()


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