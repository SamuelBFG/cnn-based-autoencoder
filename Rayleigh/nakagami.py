from scipy.stats import nakagami
import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(1, 1)

mu = 1

rv = nakagami(mu, scale=2)

x = np.linspace(0, 2, 100)

ax.plot(x, nakagami.pdf(x, mu), 'r-', lw=1, alpha=0.6, label='nakagami pdf')
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
ax.legend(loc='best', frameon=False)
plt.show()

mean, var = nakagami.stats(mu, moments='mv')