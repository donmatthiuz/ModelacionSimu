import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


def lcg(a,c,m,seed,n):
x = np.empty(n,dtype=np.int64)
xi = seed
for i in range(n):
xi = (a*xi + c) % m
x[i] = xi
return x


os.makedirs('../imagenes', exist_ok=True)


N = 100000


# conjunto 1 (Park-Miller)
a1,c1,m1,seed1 = 16807,0,2147483647,12345
x1 = lcg(a1,c1,m1,seed1,N)
u1 = x1.astype(np.float64)/m1
mean1,var1 = u1.mean(),u1.var()
ks1 = stats.kstest(u1,'uniform')
plt.hist(u1,bins=50)
plt.savefig('../imagenes/problema1_hist_set1.png')
# plt.show()


# conjunto 2 (glibc-like)
a2,c2,m2,seed2 = 1103515245,12345,2147483648,54321
x2 = lcg(a2,c2,m2,seed2,N)
u2 = x2.astype(np.float64)/m2
mean2,var2 = u2.mean(),u2.var()
ks2 = stats.kstest(u2,'uniform')
plt.clf()
plt.hist(u2,bins=50)
plt.savefig('../imagenes/problema1_hist_set2.png')
# plt.show()


print('SET1 mean,var,ks.statistic,ks.pvalue', mean1, var1, ks1.statistic, ks1.pvalue)
print('SET2 mean,var,ks.statistic,ks.pvalue', mean2, var2, ks2.statistic, ks2.pvalue)
