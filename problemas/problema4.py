import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

os.makedirs('../imagenes', exist_ok=True)

p = 0.2
N = 100000
seed = 12345

rng = np.random.default_rng(seed)
theo_rv = stats.geom(p)
theoretical = theo_rv.rvs(size=N, random_state=seed)

u = rng.random(N)
empirical = (np.floor(np.log(1 - u) / np.log(1 - p)).astype(int) + 1)

cdf_target = 0.999
K = max(10, int(np.ceil(theo_rv.ppf(cdf_target))))
vals = np.arange(1, K + 1)
pmf = theo_rv.pmf(vals)
exp_counts = (pmf * N).tolist()
obs_counts = [np.sum(empirical == k) for k in vals]
tail_exp = N * (1 - theo_rv.cdf(K))
tail_obs = np.sum(empirical > K)
exp_counts.append(tail_exp)
obs_counts.append(tail_obs)

while any(e < 5 for e in exp_counts) and len(exp_counts) > 1:
    exp_counts[-2] += exp_counts[-1]
    obs_counts[-2] += obs_counts[-1]
    exp_counts.pop()
    obs_counts.pop()

chisq_stat, chisq_p = stats.chisquare(f_obs=np.array(obs_counts), f_exp=np.array(exp_counts))

ks_stat, ks_p = stats.ks_2samp(theoretical, empirical)

alpha = 0.05

print(f"N={N}, p={p}")
print(f"Chi-square: stat={chisq_stat:.6f}, p={chisq_p:.6f}")
print(f"KS 2-sample: stat={ks_stat:.6f}, p={ks_p:.6f}")

max_plot = 50
vals_plot = np.arange(1, max_plot + 1)
obs_theo = np.bincount(theoretical, minlength=max_plot+1)[1:max_plot+1]
obs_emp = np.bincount(empirical, minlength=max_plot+1)[1:max_plot+1]
obs_theo = obs_theo / obs_theo.sum()
obs_emp = obs_emp / obs_emp.sum()

x = vals_plot - 0.4
width = 0.4
plt.bar(x, obs_theo, width=width, label='theoretical (scipy)', align='edge')
plt.bar(x + width, obs_emp, width=width, label='empirical (inverse CDF)', align='edge')
plt.xlabel('k')
plt.ylabel('relative freq')
plt.legend()
plt.tight_layout()
plt.savefig('../imagenes/problema4_geom_compare.png')
plt.clf()
#plot()
