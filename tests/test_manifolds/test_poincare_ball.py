import warnings

import autograd.numpy as np
from numpy import linalg as la, random as rnd, testing as np_testing

from pymanopt.manifolds import (Sphere, SphereSubspaceComplementIntersection,
                                SphereSubspaceIntersection)
from pymanopt.tools import testing
from .._test import TestCase


class TestSphereManifold(TestCase):
    def setUp(self):
        self.m = m = 100
        self.n = n = 50
        self.man = Sphere(m, n)

        # For automatic testing of ehess2rhess
        self.proj = lambda x, u: u - np.tensordot(x, u, np.ndim(u)) * x

    def test_dim(self):
        pass
        # assert self.man.dim == self.m * self.n - 1

    def test_typicaldist(self):
        pass
        # np_testing.assert_almost_equal(self.man.typicaldist, np.pi)

    def test_dist(self):
        pass
        # s = self.man
        # x = s.rand()
        # y = s.rand()
        # correct_dist = np.arccos(np.tensordot(x, y))
        # np_testing.assert_almost_equal(correct_dist, s.dist(x, y))

    def test_inner(self):
        pass
        # s = self.man
        # x = s.rand()
        # u = s.randvec(x)
        # v = s.randvec(x)
        # np_testing.assert_almost_equal(np.sum(u * v), s.inner(x, u, v))

    def test_proj(self):
        pass
        #  Construct a random point X on the manifold.
        # X = rnd.randn(self.m, self.n)
        # X /= la.norm(X, "fro")

        #  Construct a vector H in the ambient space.
        # H = rnd.randn(self.m, self.n)

        #  Compare the projections.
        # np_testing.assert_array_almost_equal(H - X * np.trace(X.T.dot(H)),
        #                                      self.man.proj(X, H))

    def test_egrad2rgrad(self):
        pass
        # Should be the same as proj
        #  Construct a random point X on the manifold.
        # X = rnd.randn(self.m, self.n)
        # X /= la.norm(X, "fro")

        #  Construct a vector H in the ambient space.
        # H = rnd.randn(self.m, self.n)

        #  Compare the projections.
        # np_testing.assert_array_almost_equal(H - X * np.trace(X.T.dot(H)),
        #                                      self.man.egrad2rgrad(X, H))

    def test_ehess2rhess(self):
        pass
        # x = self.man.rand()
        # u = self.man.randvec(x)
        # egrad = rnd.randn(self.m, self.n)
        # ehess = rnd.randn(self.m, self.n)

        # np_testing.assert_allclose(testing.ehess2rhess(self.proj)(x, egrad,
        #                                                           ehess, u),
        #                            self.man.ehess2rhess(x, egrad, ehess, u))

    def test_retr(self):
        pass
        # Test that the result is on the manifold and that for small
        # tangent vectors it has little effect.
        # x = self.man.rand()
        # u = self.man.randvec(x)

        # xretru = self.man.retr(x, u)
        # np_testing.assert_almost_equal(la.norm(xretru), 1)

        # u = u * 1e-6
        # xretru = self.man.retr(x, u)
        # np_testing.assert_allclose(xretru, x + u)

    def test_norm(self):
        pass
        # x = self.man.rand()
        # u = self.man.randvec(x)

        # np_testing.assert_almost_equal(self.man.norm(x, u), la.norm(u))

    def test_rand(self):
        pass
        # Just make sure that things generated are on the manifold and that
        # if you generate two they are not equal.
        # s = self.man
        # x = s.rand()
        # np_testing.assert_almost_equal(la.norm(x), 1)
        # y = s.rand()
        # assert np.linalg.norm(x - y) > 1e-3

    def test_randvec(self):
        # Just make sure that things generated are in the tangent space and
        # that if you generate two they are not equal.
        # s = self.man
        # x = s.rand()
        # u = s.randvec(x)
        # v = s.randvec(x)
        # np_testing.assert_almost_equal(np.tensordot(x, u), 0)

        # assert np.linalg.norm(u - v) > 1e-3

    def test_transp(self):
        pass
        # Should be the same as proj
        # s = self.man
        # x = s.rand()
        # y = s.rand()
        # u = s.randvec(x)

        # np_testing.assert_allclose(s.transp(x, y, u), s.proj(y, u))

    def test_exp_log_inverse(self):
        pass
        # s = self.man
        # X = s.rand()
        # U = s.randvec(X)
        # Uexplog = s.exp(X, s.log(X, U))
        # np_testing.assert_array_almost_equal(U, Uexplog)

    def test_log_exp_inverse(self):
        pass
        # s = self.man
        # X = s.rand()
        # U = s.randvec(X)
        # Ulogexp = s.log(X, s.exp(X, U))
        # np_testing.assert_array_almost_equal(U, Ulogexp)

    def test_pairmean(self):
        pass
        # s = self.man
        # X = s.rand()
        # Y = s.rand()
        # Z = s.pairmean(X, Y)
        # np_testing.assert_array_almost_equal(s.dist(X, Z), s.dist(Y, Z))