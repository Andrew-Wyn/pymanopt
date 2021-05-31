import math

import numpy as np
import numpy.linalg as la

from pymanopt.manifolds.manifold import Manifold


class PoincareBall(Manifold):
    def __init__(self, k, n):
        self.k = k
        self.n = n
        self.dimension = k*n
        super().__init__(
            "{} PoincareBall over R^{}".format(self.k, self.n), self.dimension,
            )

    def conformal_factor(self, X):
        return 2/(1 - np.sum(X*X, axis=0))

    def mobius_add(self, x, y):
        x_dot_y = np.sum(x*y, axis=0)
        x_norm_q = np.sum(x*x, axis=0)
        y_norm_q = np.sum(y*y, axis=0)

        num = (1 + 2*x_dot_y + y_norm_q)*x + (1 - x_norm_q)*y
        den = 1 + 2*x_dot_y + x_norm_q*y_norm_q

        return num/den

    def typicaldist(self):
        return self.dim / 8

    def inner(self, X, G, H):
        return np.sum(np.sum(G*H, axis=0) * self.conformal_factor(X)**2)

    def proj(self, X, G):
        # Identity map since the embedding space is the tangent space R^n
        return G

    def norm(self, X, G):
        return math.sqrt(self.inner(X, G, G))

    def rand(self):
        isotropic = np.random.standard_normal(self.n, self.k)
        isotropic = isotropic / la.norm(isotropic)
        radius = np.random.rand(self.k) ** (1 / self.n)
        x = isotropic * radius
        return x

    def randvec(self, X):
        v = np.random.rand(self.n, self.k)
        v = v / self.norm(X, v)
        return v

    def zerovec(self, X):
        return np.zeros(X.shape)

    def dist(self, X, Y):
        norms2x = np.sum(X*X, axis=0)
        norms2y = np.sum(Y*Y, axis=0)
        norms2diff = np.sum((X - Y)*(X - Y), axis=0)
        a = max(
            1,
            1 + 2*(norms2diff / ((1-norms2x)*(1-norms2y))),
            )
        return math.sqrt(sum(np.arccosh(a)**2))

    def egrad2rgrad(self, X, G):
        factor_q = self.conformal_factor(X)**2
        return G/factor_q

    def ehess2rhess(self, X, G, H, U):
        factor = self.conformal_factor(X)
        return (U * np.sum(G*X, axis=0) - G * np.sum(U*X, axis=0)
                - X * np.sum(U*G, axis=0) + H/factor)/factor

    def retr(self, X, U):
        return self.exp(X, U)

    def exp(self, X, U):
        norm_u = la.norm(U)
        factor = (1 - np.sum(X*X, axis=0))
        # avoid division by 0
        tmp = np.tanh(norm_u/factor) * (U/((norm_u + (norm_u == 0))))
        return self.mobius_add(X, tmp)

    def log(self, X, Y):
        a = self.mobius_add(-X, Y)
        b = la.norm(a)
        factor = 1 - np.sum(X*X, axis=0)
        return a * factor * np.arctanh(b) / b

    def transp(self, X1, X2, G):
        return G

    def pairmean(self, X, Y):
        return self.exp(X, self.log(X, Y) / 2)
