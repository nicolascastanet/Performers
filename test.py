import torch
import numpy as np
from performer import SMapprox
import matplotlib.pyplot as plt


l = 1024
d = 16

num_samples = 15

# random feature sizes to try
ms = torch.arange(d, 200, 16)


# Experiment:
# Sin/Cos attention vs Positive attention

sincos = []
positive = []
temperature = 1.5

SM = SMapprox(num_samples,d)

np.random.seed(0)
for m in ms:
    sincos.append([])
    positive.append([])

    for _ in range(num_samples):
        q = torch.randn(l, d) * temperature
        k = torch.randn(l, d) * temperature
        v = torch.randn(l, d) * temperature

        att_true = SM.att(q, k, v)

        A_hat = SM(q,k,v)

        sincos[-1].append(SM.mse(att_true, SM(q, k, v, pos=False)))
        positive[-1].append(SM.mse(att_true, SM(q, k, v, pos=True)))

sincos = torch.tensor(sincos)
positive = torch.tensor(positive)


def plot_line(x, y, label):
    mean = y.mean(axis=1)
    std = y.std(axis=1)
    plt.plot(x, mean, label=label)
    plt.fill_between(x, mean + std, mean - std, alpha=0.1)


plt.figure(figsize=(5, 3), dpi=300)
plot_line(ms, sincos, "Sin/Cos")
plot_line(ms, positive, "Positive")
plt.yscale("log")
# plt.ylim(1e-2, 1e8)
plt.ylabel("Output MSE")
plt.xlabel("Num. Features $R$")
plt.legend()
plt.savefig("trig_vs_positive.png", bbox_inches="tight")
