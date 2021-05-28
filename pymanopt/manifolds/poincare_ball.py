import math
from random import random

import numpy as np
import numpy.linalg as la

from pymanopt.manifolds.manifold import Manifold


class PoincareBall(Manifold):
    def __init__(self, dimension):
        super().__init__(
            "PoincareBall over R^{}".format(dimension), dimension
            )

    def conformal_factor(self, x):
        return 2/(1 - np.dot(x, x))

    def mobius_add(self, x, y):
        x_dot_y = np.dot(x, y)
        x_norm_q = np.dot(x, x)
        y_norm_q = np.dot(y, y)

        num = (1 + 2*x_dot_y + y_norm_q)*x + (1 - x_norm_q)*y
        den = 1 + 2*x_dot_y + x_norm_q*y_norm_q

        if den == 0:
            den = 10e-14

        return num/den

    def typicaldist(self):
        return self.dim / 8

    def inner(self, X, G, H):
        return self.conformal_factor(X)**2*np.dot(G, H)

    def proj(self, X, G):
        # Identity map since the embedding space is the tangent space R^n
        return G

    def norm(self, X, G):
        return self.conformal_factor(X)*la.norm(G)

    def rand(self):
        isotropic = np.random.rand(self.dim)
        isotropic = isotropic / la.norm(isotropic)
        radius = random() ** (1 / self.dim)
        x = isotropic * radius
        return x

    def randvec(self, X):
        v = np.random.rand(self.dim)
        v = v / self.norm(X, v)
        return v

    def zerovec(self, X):
        return np.zeros(X.shape)

    def dist(self, X, Y):
        a = max(1, 1 + 2*(np.dot(X-Y, X-Y)
                / ((1-np.dot(X, X))*(1-np.dot(Y, Y)))))
        return np.arccosh(a)

    def egrad2rgrad(self, X, G):
        factor_q = self.conformal_factor(X)**2
        return G/factor_q

    def ehess2rhess(self, X, G, H, U):
        factor = self.conformal_factor(X)
        return (U * np.dot(G, X) - G * np.dot(U, X) - X * np.dot(U, G)
                + H/factor)/factor

    def retr(self, X, U):
        return self.exp(X, U)

    def exp(self, X, U):
        norm_u = la.norm(U)
        # avoid division by 0
        tmp = np.tanh(1/2*self.norm(X, U)) * (U/((norm_u + (norm_u == 0))))
        return self.mobius_add(X, tmp)

    def log(self, X, Y):
        a = self.mobius_add(-X, Y)
        b = math.sqrt(np.dot(a, a))
        c = 2/self.conformal_factor(X)*np.arctanh(b)
        return c*a/b

    def transp(self, X1, X2, G):
        return G

    def pairmean(self, X, Y):
        return self.exp(X, self.log(X, Y) / 2)
