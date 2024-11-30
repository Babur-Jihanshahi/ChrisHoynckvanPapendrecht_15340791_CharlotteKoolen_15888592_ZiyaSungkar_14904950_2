import numpy as np
import matplotlib.pyplot as plt

rho = np.linspace(0.70, 0.9999, 1000)
mu = 1.00
Wq_mm1 = rho / (mu * (1 - rho))
P0 = (1 + 2 * rho + ((2 * rho) ** 2) / (2 * (1 - rho))) ** (-1)
Wq_mm2 = (rho**2 * P0) / (mu * (1 - rho)**2)

# font sizes
font_title = {'size': 20}
font_labels = {'size': 20}
font_ticks = {'size': 20}
font_legend = {'size': 22}

plt.figure(figsize=(10, 6), dpi=300)
plt.plot(rho, Wq_mm1, label=r'M/M/1 Waiting Time $W_q^{M/M/1}$', color='blue')
plt.plot(rho, Wq_mm2, label=r'M/M/2 Waiting Time $W_q^{M/M/2}$', color='red', linestyle='--')
plt.xlabel(r'System Load $\rho$', fontdict=font_labels)
plt.ylabel(r'Mean Waiting Time $W_q$', fontdict=font_labels)
plt.title('Mean Waiting Time Comparison for M/M/1 and M/M/2', fontdict=font_title)
plt.xticks(fontsize=font_ticks['size'])
plt.yticks(fontsize=font_ticks['size'])
plt.legend(prop={'size': font_legend['size']})
plt.grid(True)
plt.ylim(0, 50)
plt.savefig('mm1_mm2_Wq_comparison.png', dpi=300, bbox_inches='tight')
plt.show()