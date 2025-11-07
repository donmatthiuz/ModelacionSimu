import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

os.makedirs('../imagenes', exist_ok=True)

N = 100000
# seed - elegi usar mi carnet
rng = np.random.Generator(np.random.MT19937(22779))
u = rng.random(N)
mean,var = u.mean(),u.var()
ks = stats.kstest(u,'uniform')
plt.hist(u,bins=50)
plt.savefig('../imagenes/problema2_hist_mt.png')
# plt.show()
print('MT mean,var,ks.statistic,ks.pvalue', mean, var, ks.statistic, ks.pvalue)
