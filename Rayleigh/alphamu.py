import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,30)

tru_bler = [0.1465965625, 0.12663375, 0.108449375, 0.09171625, 0.077281875, 0.0642928125, 0.05279125, 0.0434259375, 0.035626875, 0.0287071875, 0.023293125, 0.018800625, 0.0150484375, 0.012090625, 0.0097278125, 0.0077828125, 0.0061540625, 0.004979375, 0.00392625, 0.0031925, 0.00244375, 0.0020071875, 0.001640625, 0.00129125, 0.0010525, 0.000811875, 0.0006659375, 0.0005415625, 0.0004575, 0.0003690625]


plt.plot(x,tru_bler,label='BLER Rayleigh (14dB)')

plt.yscale('log')
plt.grid(True)
plt.legend(loc='upper right')
axes = plt.gca()
plt.title('Alpha-mu')
axes.set_xlim([0,30])
axes.set_ylim([10**(-6),10**(0)])
plt.show()