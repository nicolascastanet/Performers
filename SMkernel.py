import numpy as np
import torch
import torch.nn as nn

class SMapprox(nn.Module):
    def __init__(self,rd, hd, ort=True, pos=True):
        super().__init__()
        self.random_dim = rd
        self.hidden_dim = hd
        self.ortho = ort
        self.pos = pos
        self.vects = self.random_vect()
        
    def redrawn(self):
        self.vects = self.random_vect()

    # Vanila Transformer attention implementation
    def att(self, q, k, v, normalize=True):
        l, d = q.shape
        normalizer = 1 / (d ** 0.5) if normalize else 1
        a = torch.exp(q @ k.T * normalizer)
        d_inv = torch.diag(1 / (a @ torch.ones(l)))
        return d_inv @ a @ v


    # Perfomer attention implementation using some random feature map phi
    def att_hat(self,q, k, v, phi, normalize=True):

        if len(q.shape) == 2:
            l, d = q.shape
            normalizer = 1 / (d ** 0.25)
            q_prime = phi(q * normalizer)
            k_prime = phi(k * normalizer)
            d_inv = torch.diag(1 / (q_prime @ (k_prime.T @ torch.ones(l))))
            return d_inv @ (q_prime @ (k_prime.T @ v))


        # Batch mode
        elif len(q.shape) == 3:
            batch_size, l, d = q.shape
            normalizer = 1 / (d ** 0.25)
            out = []

            for i in range(batch_size):

                q_prime = phi(q[i] * normalizer)
                k_prime = phi(k[i] * normalizer)
                d_inv = torch.diag(1 / (q_prime @ (k_prime.T @ torch.ones(l))))
                res = d_inv @ (q_prime @ (k_prime.T @ v[i]))
                out.append(res)

            return torch.stack(out)


    # random feature map
    def phi(self,h, fs):
        return lambda x: (
            h(x)
            / (self.random_dim**1/2)
            * torch.cat(
                [f(torch.einsum("...d,md->...m", x, self.vects)) for f in fs],
                axis=-1,
            )
        )


    # Performer "sin/cos" attention
    def sincos_att_hat(self,q, k, v, normalize=True):
        def h(x):
            return torch.exp(torch.square(x).sum(axis=-1, keepdims=True) / 2)

        sin = lambda x: torch.sin(2 * np.pi * x)
        cos = lambda x: torch.cos(2 * np.pi * x)

        kernel = self.phi(h, [sin, cos])
        return self.att_hat(q, k, v, kernel, normalize)


    # Performer "positive" attention
    def positive_att_hat(self,q, k, v, normalize=True):
        def h(x):
            return torch.exp(-torch.square(x).sum(axis=-1, keepdims=True) / 2)

        kernel = self.phi(h, [torch.exp])
        return self.att_hat(q, k, v, kernel, normalize)


    # generate IID Gaussian random features
    def iid_gaussian(self,m, d):
        return torch.randn(size=(m, d))


    # generate orthogonal Gaussian random features
    def orthogonal_gaussian(self,m, d):
        def orthogonal_square():
            # create orthogonal square matrix using Gram-Schmidt
            q, _ = torch.qr(self.iid_gaussian(d, d))
            return q.T

        num_squares = int(m / d)
        blocks = [orthogonal_square() for _ in range(num_squares)]

        remainder = m - d * num_squares
        if remainder:
            blocks.append(orthogonal_square()[:remainder])

        matrix = torch.cat(blocks)
        matrix /= (num_squares + remainder / d)**(1/2)

        return matrix

    def random_vect(self):
        if self.ortho:
            return self.orthogonal_gaussian(self.random_dim, self.hidden_dim)
        else :
            return self.iid_gaussian(self.random_dim, self.hidden_dim)


    # mean squared error
    def mse(self,a, b):
        return torch.square(a - b).mean()


    def forward(self,q,k,v,pos=True):
        
        if pos:
            return self.positive_att_hat(q,k,v,self.random_vect)
        else:
            return self.sincos_att_hat(q,k,v,self.random_vect)

