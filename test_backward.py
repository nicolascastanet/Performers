import torch
from SMkernel import SMapprox
import numpy as np

l = 1024
e_dim = 5
d = 16
b = 100

num_samples = 10

SM = SMapprox(num_samples,d)

# input
x = torch.randn((b,l,e_dim), requires_grad=True)

# weights
Wq = torch.randn((e_dim, d), requires_grad=True)
Wk = torch.randn((e_dim, d), requires_grad=True)
Wv = torch.randn((e_dim, d), requires_grad=True)

# querys, keys, values
Q = torch.matmul(x,Wq)
K = torch.matmul(x,Wk)
V = torch.matmul(x,Wv)

A_hat = SM(Q,K,V)

A_hat.sum().backward()
print('grad Wq',Wq.grad)



