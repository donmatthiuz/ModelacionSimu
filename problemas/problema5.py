import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

os.makedirs('../imagenes', exist_ok=True)

mu = 0.0
sigma = 1.0
N = 100000
seed = 12345

rng = np.random.default_rng(seed)
theoretical = stats.norm.rvs(loc=mu, scale=sigma, size=N, random_state=seed)
eps = 1e-12
u = rng.uniform(eps, 1 - eps, size=N)
empirical = stats.norm.ppf(u, loc=mu, scale=sigma)

bins = min(50, max(5, N // 5000))
quantiles = np.linspace(0.0, 1.0, bins + 1)
quantiles[0] = eps
quantiles[-1] = 1.0 - eps
edges = stats.norm.ppf(quantiles, loc=mu, scale=sigma)

obs_emp, _ = np.histogram(empirical, bins=edges)
expected = np.ones(bins) * (N / bins)
chisq_stat, chisq_p = stats.chisquare(f_obs=obs_emp, f_exp=expected)

ks_stat, ks_p = stats.ks_2samp(theoretical, empirical)

alpha = 0.05

print(f"N={N}, mu={mu}, sigma={sigma}")
print(f"Chi-square: stat={chisq_stat:.6f}, p={chisq_p:.6f}")
print(f"KS 2-sample: stat={ks_stat:.6f}, p={ks_p:.6f}")

plt.hist(theoretical, bins=edges, density=True, histtype='step', label='theoretical (scipy)')
plt.hist(empirical, bins=edges, density=True, alpha=0.5, label='empirical (inverse CDF)')
plt.legend()
plt.xlabel('x')
plt.ylabel('density')
plt.tight_layout()
plt.savefig('../imagenes/problema_normal_compare.png')
plt.clf()
