from numpy import linalg as la, random as rnd, testing as np_testing
import warnings
import math
import autograd.numpy as np
from pymanopt.manifolds import hyperboloid
from pymanopt.tools import testing
from .._test import TestCase


class TestHyperboloidManifold(TestCase):
    def setUp(self):
        self.n = n = 50
        self.man = hyperboloid.Hyperboloid(n)

    def _minkowski_dot(self, x, y):
        return np.dot(x[:-1], y[:-1]) - x[-1]*y[-1]

    # For automatic testing of ehess2rhess
    def _proj(self, x, u):
        return u + self._minkowski_dot(x, u) * x

    def test_dim(self):
        self.man.dim = self.n + 1
        # assert self.man.dim == self.m * self.n - 1

    def test_typicaldist(self):
        np_testing.assert_almost_equal(self.man.typicaldist, math.sqrt(self.dim))

    def test_dist(self):
        
        h = self.man
        x = h.rand()
        y = h.rand()
        correct_dist = np.arccos(-self._minkowski_dot(x, y))
        np_testing.assert_almost_equal(correct_dist, h.dist(x, y))

    def test_inner(self):
        
        h = self.man
        x = h.rand()
        u = h.randvec(x)
        v = h.randvec(x)
        np_testing.assert_almost_equal(self._minkowski_dot(u, v), h.inner(x, u, v))

    def test_proj(self):
        #  Construct a random point X on the manifold.
        x = self.man.rand()

        #  Construct a vector H in the ambient space.
        h = rnd.randn(self.n)

        #  Compare the projections.
        np_testing.assert_array_almost_equal(self._proj(x, h),
                                             self.man.proj(x, h))

    def test_egrad2rgrad(self):
        pass
        #  Should be the same as proj
        #  Construct a random point X on the manifold.
        X = self.man.rand()

        #  Construct a vector H in the ambient space.
        H = rnd.randn(self.n+1)
        
        H[-1] = -H[-1]

        #  Compare the projections.
        np_testing.assert_array_almost_equal(self._proj(X, H),
                                             self.man.egrad2rgrad(X, H))

    def test_ehess2rhess(self):
        pass
        # x = self.man.rand()
        # u = self.man.randvec(x)
        # egrad = rnd.randn(self.n)
        # ehess = rnd.randn(self.m, self.n)

        # egrad[-1] = -egrad[-1]
        # ehess[:, -1] = -ehess[:, -1]

        # np_testing.assert_allclose(testing.ehess2rhess(self._proj)(x, egrad,
        #                                                           ehess, u),
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
        x = self.man.rand()
        u = self.man.randvec(x)

        np_testing.assert_almost_equal(self.man.norm(x, u), math.sqrt(max(0, self._minkowski_dot(u, u))))

    def test_rand(self):
        # Just make sure that things generated are on the manifold and that
        # if you generate two they are not equal.
        h = self.man
        x = h.rand()
        np_testing.assert_almost_equal(self._minkowski_dot(x, x), -1)
        y = h.rand()
        assert np.linalg.norm(x - y) > 1e-3

    def test_randvec(self):
        # Just make sure that things generated are in the tangent space and
        # that if you generate two they are not equal.
        h = self.man
        x = h.rand()
        u = h.randvec(x)
        v = h.randvec(x)
        np_testing.assert_almost_equal(self._minkowski_dot(x, u), 0)

        assert np.linalg.norm(u - v) > 1e-3

    def test_transp(self):
        pass
        # Should be the same as proj
        # s = self.man
        # x = s.rand()
        # y = s.rand()
        # u = s.randvec(x)

        # np_testing.assert_allclose(s.transp(x, y, u), s.proj(y, u))

    def test_exp_log_inverse(self):
        h = self.man
        x = h.rand()
        u = h.rand()
        Uexplog = h.exp(x, h.log(x, u))
        np_testing.assert_array_almost_equal(u, Uexplog)

    def test_log_exp_inverse(self):
        h = self.man
        x = h.rand()
        y = h.rand()
        ylogexp = h.log(x, h.exp(x, y))
        np_testing.assert_array_almost_equal(y, ylogexp)

    def test_pairmean(self):
        h = self.man
        x = h.rand()
        y = h.rand()
        z = h.pairmean(x, y)
        np_testing.assert_array_almost_equal(h.dist(x, z), h.dist(y, z))