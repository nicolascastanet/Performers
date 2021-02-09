import numpy as np

def z_positive(x, omega):
    coef = np.exp(-np.square(x).sum(axis=-1, keepdims=True) / 2)
    product = np.einsum("...d,rd->...r", x, omega)
    return coef * np.exp(product)

def z_sin_cos(x, omega):
    sin = lambda x: np.sin(2 * np.pi * x)
    cos = lambda x: np.cos(2 * np.pi * x)

    coef = np.exp(np.square(x).sum(axis=-1, keepdims=True) / 2)
    product = np.einsum("...d,rd->...r", x, omega)
    return coef * np.concatenate([sin(product), cos(product)], axis=-1)

def attention_hat(q, k, v, random_dim, ortho=True, pos=True):
    l, d = q.shape
    normalizer = 1 / (d ** 0.25)               # to normalize before multiplication
    if ortho:
        omega = orthogonal_gaussian(random_dim, d)
    else:
        omega = np.random.randn(random_dim, d)     # generate i.i.d. gaussian features

    if pos:
        q_prime = z_positive(q * normalizer, omega) # apply feature map z to Q
        k_prime = z_positive(k * normalizer, omega) # apply feature map z to K
    else:
        q_prime = z_sin_cos(q * normalizer, omega) # apply feature map z to Q
        k_prime = z_sin_cos(k * normalizer, omega) # apply feature map z to K
    # rest of attention (note the order of operations is changed for efficiency)
    d_inv = np.diag(1 / (q_prime @ (k_prime.T @ np.ones(l))))
    return d_inv @ (q_prime @ (k_prime.T @ v))


def iid_gaussian(m, d):
    return np.random.normal(size=(m, d))


# generate orthogonal Gaussian random features
def orthogonal_gaussian(m, d):
    def orthogonal_square():
        # create orthogonal square matrix using Gram-Schmidt
        q, _ = np.linalg.qr(iid_gaussian(d, d))
        return q.T

    num_squares = int(m / d)
    blocks = [orthogonal_square() for _ in range(num_squares)]

    remainder = m - d * num_squares
    if remainder:
        blocks.append(orthogonal_square()[:remainder])

    matrix = np.vstack(blocks)
    matrix /= np.sqrt(num_squares + remainder / d)
    # matrix = np.diag(np.sqrt(d) * np.ones(m)) @ matrix

    return matrix

