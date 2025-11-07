import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

os.makedirs('../imagenes', exist_ok=True)

N = 100000
rng = np.random.Generator(np.random.MT19937(22779))
u = rng.random(N)

mean = u.mean()
var = u.var(ddof=0)
theoretical_mean = 0.5
theoretical_var = 1.0 / 12.0
se_mean = np.sqrt(theoretical_var / N)
z_mean = (mean - theoretical_mean) / se_mean
p_z_mean = 2 * (1 - stats.norm.cdf(abs(z_mean)))

ks = stats.kstest(u, 'uniform')
bins = 50
counts, _ = np.histogram(u, bins=bins, range=(0.0, 1.0))
expected_counts = np.ones(bins) * (N / bins)
chisq, p_chisq = stats.chisquare(counts, f_exp=expected_counts)
autocorr = np.corrcoef(u[:-1], u[1:])[0, 1]

plt.hist(u, bins=bins)
plt.savefig('../imagenes/problema2_hist_mt.png')
# plt.show()
plt.clf()

plt.scatter(u[:-1], u[1:], s=1)
plt.xlabel('u_i')
plt.ylabel('u_{i+1}')
plt.savefig('../imagenes/problema2_scatter_mt.png')
# plt.show()
plt.clf()

print('MT')
print('N, mean, var', N, mean, var)
print('theoretical mean,var', theoretical_mean, theoretical_var)
print('z for mean, p-value', z_mean, p_z_mean)
print('KS statistic, p-value', ks.statistic, ks.pvalue)
print('Chi-square, p-value', chisq, p_chisq)
print('lag-1 autocorrelation', autocorr)
