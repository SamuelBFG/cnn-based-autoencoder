import numpy as np
from numpy import sqrt
from numpy.random import rand, randn, normal
from scipy.special import erf, erfc
import matplotlib.pyplot as plt
import random
import time
from scipy.integrate import quad



###################
#### SIMULATED ####
###################  
N = 5000000
EbNodB = np.arange(0,36,1)
ber_awgn = [None]*len(EbNodB)
ber_ray = [None]*len(EbNodB)


for n in range (0, len(EbNodB)): 
	EbNo=10.0**(EbNodB[n]/10.0)
	noise_std = 1/sqrt(2*EbNo)
	noise_mean = 0
	no_errors = 0

	# Transmitted N symbols
	tx_symb = 2 * (rand(N) >= 0.5) - 1

	ch_coeff = sqrt(normal(0,1,N)**2 + normal(0,1,N)**2)/sqrt(2)

	# y = h*x + n
	rx_symb_awgn = tx_symb + noise_std * randn(N)
	rx_symb_ray = tx_symb*ch_coeff + noise_std * randn(N)

	# Decision
	d_awgn = 2 * (rx_symb_awgn >= 0) - 1
	d_ray = 2 * (rx_symb_ray >= 0) - 1


	errors_awgn = (tx_symb != d_awgn).sum()
	errors_ray = (tx_symb != d_ray).sum()

	ber_awgn[n] = 1.0 * errors_awgn / N
	ber_ray[n] = 1.0 * errors_ray / N

    
	print("EbNodB:", EbNodB[n]) 
	# print ("Error bits:", errors_awgn)
	print ("AWGN Error probability:", ber_awgn[n])
	print ("RAYLEIGH Error probability:", ber_ray[n])



        
###################
### THEORETICAL ###
###################
ber_awgn_th = 0.5*erfc(np.sqrt(10**(EbNodB/10)))

def rayleigh(i, x):
	A = 0.5*erfc(np.sqrt(10**(x/10)))
	B = (1/i) * np.exp(-x/i)
	return A*B



ber_ray_th = [None]*len(EbNodB)
for i in EbNodB:
	ber_ray_th[i], _ = quad(rayleigh, 0, np.inf, args=i)

print(ber_ray_th)

# test = sum(ber_ray_th) / len(ber_ray_th)

plt.plot(EbNodB, ber_awgn, 'ro', label='AWGN - Simulated')
plt.plot(EbNodB, ber_ray, 'bo', label='RAYLEIGH - Simulated')
plt.plot(EbNodB, ber_awgn_th, 'r', label='AWGN - Theoretical')
plt.plot(EbNodB, ber_ray_th, 'b', label='RAYLEIGH - Theoretical')
plt.axis([0, 35, 1e-5, 0.1])
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('EbNo(dB)')
plt.ylabel('BER')
plt.legend(loc='upper right')
plt.grid(True)
plt.title('BPSK Modulation')
plt.show()