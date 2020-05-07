from numpy import sqrt
import random
import matplotlib.pyplot as plt
  
N = 100000
EbNodB_range = range(0, 36)
ber = [None]*len(EbNodB_range)
tx_symbol = 0
noise = 0 
ch_coeff = 0
rx_symbol = 0 
det_symbol = 0 

for n in range (0, len(EbNodB_range)): 
 
    EbNodB = EbNodB_range[n]   
    EbNo=10.0**(EbNodB/10.0)
    noise_std = 1/sqrt(2*EbNo)
    noise_mean = 0
    
    no_errors = 0

    tx_symbol = 2*random.randint(0,N)-1
    noise = random.gauss(noise_mean, noise_std)
    ch_coeff = sqrt(random.gauss(0,1)**2+random.gauss(0,1)**2)/sqrt(2)
    rx_symbol = tx_symbol*ch_coeff + noise
    det_symbol = 2 * (rx_symbol >= 0) - 1
    no_errors += 1*(tx_symbol != det_symbol)  
    
    ber[n] = no_errors / N
    print ("EbNodB:", EbNodB)
    print ("Numbder of errors:", no_errors)
    print ("Error probability:", ber[n] )
        
plt.plot(EbNodB_range, ber, 'bo-')
plt.axis([0, 35, 1e-5, 0.1])
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('EbNo(dB)')
plt.ylabel('BER')
plt.grid(True)
plt.title('BPSK Modulation')
plt.show()