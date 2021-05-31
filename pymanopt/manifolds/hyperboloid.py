import math

import numpy as np

from pymanopt.manifolds.manifold import Manifold


class Hyperboloid(Manifold):
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.dimension = n * k
        super().__init__(
            "{} Hyperboloid over R^{}:1".format(k, n), self.dimension,
            )

    def _squeeze(self, X):
        if self.k == 1 and len(X.shape) > 1:
            return np.squeeze(X, axis=1)
        else:
            return X

    def _pack(self, X):
      if len(X.shape) == 1:
        return np.expand_dims(X, axis=1)
      else:
        return X

    def inner_minkowski_columns(self, U, V):
        U = self._pack(U)
        V = self._pack(V)
        return np.array(
              [
                  np.dot(U[:, i][:-1], V[:, i][:-1]) - U[:, i][-1]*V[:, i][-1]
                  for i in range(self.k)
              ]
              )

    def typicaldist(self):
        return math.sqrt(self.dim)

    def inner(self, X, U, V):
        U = self._pack(U)
        V = self._pack(V)
        return np.sum(self.inner_minkowski_columns(U, V))

    def proj(self, X, G):
        X = self._pack(X)
        G = self._pack(G)
        inners = self.inner_minkowski_columns(X, G)
        return self._squeeze(G + X*inners)

    def norm(self, X, G):
        return math.sqrt(max(0, self.inner(G, G)))

    def rand(self):
        ret = np.zeros((self.n+1, self.k))
        x0 = np.random.normal(size=(self.n, self.k))
        x1 = np.sqrt(1 + np.sum(x0 * x0, axis=0))
        ret[:-1, :] = x0
        ret[-1, :] = x1
        return self._squeeze(ret)

    def randvec(self, X):
        X = self._pack(X)
        U = self.proj(X, np.random.rand(X.shape))
        return self._squeeze(U / self.norm(X, U))

    def zerovec(self, X):
        return np.zeros(X.shape)

    def _dists(self, X, Y):
        X = self._pack(X)
        Y = self._pack(Y)
        alpha = -self.inner_minkowski_columns(X, Y)
        alpha[alpha < 1] = 1
        return np.arccosh(alpha)

    def dist(self, X, Y):
        X = self._pack(X)
        Y = self._pack(Y)
        return la.norm(self._dists(X, Y))

    def egrad2rgrad(self, X, G):
        X = self._pack(X)
        G = self._pack(G)
        G[-1, :] = -G[-1, :]
        return self._squeeze(self.proj(X, G))

    def ehess2rhess(self, X, G, H, U):
        X = self._pack(X)
        G = self._pack(G)
        H = self._pack(H)
        U = self._pack(U)
        G[-1, :] = -G[-1, :]
        H[-1, :] = -H[-1, :]
        inners = self.inner_minkowski_columns(X, G)
        return self._squeeze(self.proj(X, U*inners + H))

    def retr(self, X, U):
        X = self._pack(X)
        U = self._pack(U)
        return self._squeeze(self.exp(X, U))

    def exp(self, X, U):
        X = self._pack(X)
        U = self._pack(U)
        # compute the individual minkowski norm for each individual column of U
        mink_inners = self.inner_minkowski_columns(U, U)
        vnormmf = np.vectorize(lambda x: math.sqrt(max(0, x)))
        mink_norms = vnormmf(mink_inners)
        a = np.sinh(mink_norms)/mink_norms
        a[np.isnan(a)] = 1
        return self._squeeze(np.cosh(mink_norms)*X + U*a)

    def log(self, X, Y):
        X = self._pack(X)
        Y = self._pack(Y)
        d = self._dists(X, Y)
        a = d/np.sinh(d)
        a[np.isnan(a)] = 1
        return self._squeeze(self.proj(X, Y*a))

    def transp(self, X1, X2, G):
        X1 = self._pack(X1)
        X2 = self._pack(X2)
        G = self._pack(G)
        return self._squeeze(self.proj(X2, G))

    def pairmean(self, X, Y):
        X = self._pack(X)
        Y = self._pack(Y)
        return self._squeeze(self.exp(X, self.log(X, Y), 1/2))
