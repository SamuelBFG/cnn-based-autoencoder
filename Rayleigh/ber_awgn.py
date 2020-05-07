import numpy as np
from numpy.random import rand, randn
from scipy.special import erf, erfc
import matplotlib.pyplot as plt
import time

snr = np.arange(0,11,.5)
print(snr)

ber_awgn = 0.5*erfc(np.sqrt(10**(snr/10)))
ber_awgn2 = erf(np.sqrt(2*10**(snr/10)))

from numpy import sqrt
from numpy.random import rand, randn
import matplotlib.pyplot as plt
  
N = 5000000
EbNodB = np.arange(0,36,0.5)
ber_awgn = [None]*len(EbNodB)

# for n in range (0, len(EbNodB)): 

# 	print(n)
# 	time.sleep(.5)
# 	print('EbNodB', EbNodB[n])
# time.sleep(50)

for n in range (0, len(EbNodB)): 

	EbNo=10.0**(EbNodB[n]/10.0)
	x = 2 * (rand(N) >= 0.5) - 1
	noise_std = 1/sqrt(2*EbNo)
	y = x + noise_std * randn(N)
	y_d = 2 * (y >= 0) - 1
	errors = (x != y_d).sum()
	ber_awgn[n] = 1.0 * errors / N
    
	print("EbNodB:", EbNodB[n]) 
	print ("Error bits:", errors)
	print ("Error probability:", ber_awgn[n])
        
plt.plot(EbNodB, ber_awgn, 'bo')
plt.axis([0, 10, 1e-6, 0.1])
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('EbNo(dB)')
plt.ylabel('BER')
plt.grid(True)
plt.title('BPSK Modulation')
plt.show()