import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def lcg(a, c, m, seed, n):
    x = np.empty(n, dtype=np.int64)
    xi = np.int64(seed)
    for i in range(n):
        xi = (a * xi + c) % m
        x[i] = xi
    return x

os.makedirs('../imagenes', exist_ok=True)

N = 100000

# conjunto 1 (Park-Miller)
a1, c1, m1, seed1 = 16807, 0, 2147483647, 12345
x1 = lcg(a1, c1, m1, seed1, N)
u1 = x1.astype(np.float64) / m1

mean1 = u1.mean()
var1 = u1.var(ddof=0)
theoretical_mean = 0.5
theoretical_var = 1.0 / 12.0
se_mean = np.sqrt(theoretical_var / N)
z_mean1 = (mean1 - theoretical_mean) / se_mean
p_z_mean1 = 2 * (1 - stats.norm.cdf(abs(z_mean1)))

ks1 = stats.kstest(u1, 'uniform')
bins = 50
counts1, _ = np.histogram(u1, bins=bins, range=(0.0, 1.0))
expected_counts1 = np.ones(bins) * (N / bins)
chisq1, p_chisq1 = stats.chisquare(counts1, f_exp=expected_counts1)

autocorr1 = np.corrcoef(u1[:-1], u1[1:])[0, 1]

plt.hist(u1, bins=bins)
plt.savefig('../imagenes/problema1_hist_set1.png')
# plt.show()
plt.clf()

plt.scatter(x1[:-1] / m1, x1[1:] / m1, s=1)
plt.xlabel('u_i')
plt.ylabel('u_{i+1}')
plt.savefig('../imagenes/problema1_scatter_set1.png')
# plt.show()
plt.clf()

print('SET1 (Park-Miller)')
print('N, mean, var', N, mean1, var1)
print('theoretical mean,var', theoretical_mean, theoretical_var)
print('z for mean, p-value', z_mean1, p_z_mean1)
print('KS statistic, p-value', ks1.statistic, ks1.pvalue)
print('Chi-square, p-value', chisq1, p_chisq1)
print('lag-1 autocorrelation', autocorr1)
print('--------------------------------------------------')

# conjunto 2 (glibc-like)
a2, c2, m2, seed2 = 1103515245, 12345, 2147483648, 54321
x2 = lcg(a2, c2, m2, seed2, N)
u2 = x2.astype(np.float64) / m2

mean2 = u2.mean()
var2 = u2.var(ddof=0)
se_mean2 = np.sqrt(theoretical_var / N)
z_mean2 = (mean2 - theoretical_mean) / se_mean2
p_z_mean2 = 2 * (1 - stats.norm.cdf(abs(z_mean2)))

ks2 = stats.kstest(u2, 'uniform')
counts2, _ = np.histogram(u2, bins=bins, range=(0.0, 1.0))
expected_counts2 = np.ones(bins) * (N / bins)
chisq2, p_chisq2 = stats.chisquare(counts2, f_exp=expected_counts2)

autocorr2 = np.corrcoef(u2[:-1], u2[1:])[0, 1]

plt.hist(u2, bins=bins)
plt.savefig('../imagenes/problema1_hist_set2.png')
# plt.show()
plt.clf()

plt.scatter(x2[:-1] / m2, x2[1:] / m2, s=1)
plt.xlabel('u_i')
plt.ylabel('u_{i+1}')
plt.savefig('../imagenes/problema1_scatter_set2.png')
# plt.show()
plt.clf()

print('SET2 (glibc-like)')
print('N, mean, var', N, mean2, var2)
print('theoretical mean,var', theoretical_mean, theoretical_var)
print('z for mean, p-value', z_mean2, p_z_mean2)
print('KS statistic, p-value', ks2.statistic, ks2.pvalue)
print('Chi-square, p-value', chisq2, p_chisq2)
print('lag-1 autocorrelation', autocorr2)

