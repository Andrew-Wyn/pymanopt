import math

import numpy as np

from pymanopt.manifolds.manifold import Manifold


class Hyperboloid(Manifold):
    def __init__(self, dimension):
        super().__init__(self, "Hyperboloid", dimension)

    def _minkowski_dot(self, X, Y):
        return np.dot(X[:-1], Y[:-1]) - X[-1]*Y[-1]

    def typicaldist(self):
        return math.sqrt(self.dim)

    def inner(self, X, G, H):
        return self._minkowski_dot(G, H)

    def proj(self, X, G):
        inner = self._minkowski_dot(X, G)
        return G + X*inner

    def norm(self, X, G):
        return math.sqrt(max(0, self._minkowski_dot(G, G)))

    def rand(self):
        ret = np.zeros(self.dim+1)
        x0 = np.random.normal(self.dim)
        x1 = math.sqrt(1 + np.dot(x0))
        ret[:-1] = x0
        ret[-1] = x1
        return ret

    def randvec(self, X):
        U = self.proj(X, np.random.rand(X.shape))
        return U / self.norm(X, U)

    def zerovec(self, X):
        return np.zeros(X.shape)

    def dist(self, X, Y):
        alpha = max(1, -self._minkowski_dot(X, Y))
        return np.arccosh(alpha)

    def egrad2rgrad(self, X, G):
        G[-1] = -G[-1]
        return self.proj(X, G)

    def ehess2rhess(self, X, G, H, U):
        G[-1] = -G[-1]
        H[:, -1] = -H[:, -1]
        inners = self._minkowski_dot(X, G)
        return self.proj(X, U*inners + H)

    def retr(self, X, U):
        return self.exp(X, U)

    def exp(self, X, U):
        mink_norm_u = self.norm(X, U)
        a = np.sinh(mink_norm_u)/mink_norm_u
        if mink_norm_u == 0:
            a = 1
        return np.cosh(mink_norm_u)*X + U*a

    def log(self, X, Y):
        alpha = max(1, -self._minkowski_dot(X, Y))
        if alpha == 1:
            a = 1
        else:
            a = np.arccosh(alpha)/((alpha**2 - 1)**(1/2))
        return (a)*(Y - alpha*X)

    def transp(self, X1, X2, G):
        return self.proj(X2, G)

    def pairmean(self, X, Y):
        return self.exp(X, self.log(X, Y), 1/2)
