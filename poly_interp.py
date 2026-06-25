import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import root, root_scalar

import torch
def polymatrix2d(x, y, ij):

    size = len(x)
    G = torch.zeros((size, len(ij)))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x ** i * y ** j

    return G


def vec_polyfit2d(x, y, z, ij):

    # x and y are vectors of length n_images
    # z is an array of size (9, n_images)
    # ij is the polynomial's degrees for every combination of x and y

    # m is of shape (n_coeffs, n_images)
    G = polymatrix2d(x, y, ij)
    m, _, _, _ = torch.linalg.lstsq(G, z, rcond=1e-10)
    return m


class PytorchPolyFit2D(torch.nn.Module):
    def __init__(self, X, images, order=2):
        super(PytorchPolyFit2D, self).__init__()

        self.x, self.y = X
        self.images = images

        self.order = order
        self.ij = np.array([_ for _ in itertools.product(range(order + 1), range(order + 1))])

        """
        dz/dx = a3 + a4y + a5y^2 + 2a6x + 2a7xy + 2a8xy^2
        dz/dy = a1 + 2a2y + a4x + 2a5xy + a7x^2 + 2a8x^2y
        """

        # compute the polynomial's gradient
        self.gradx = self.ij[:, 0] - 1
        self.gradx[self.gradx < 0] = 0
        self.gradx = np.vstack([self.gradx, self.ij[:, 1]]).T
        self.gradmaskx = torch.tensor((self.ij[:, 0] > 0) * self.ij[:, 0])

        self.grady = self.ij[:, 1] - 1
        self.grady[self.grady < 0] = 0
        self.grady = np.vstack([self.ij[:, 0], self.grady]).T
        self.gradmasky = torch.tensor((self.ij[:, 1] > 0) * self.ij[:, 1])

        """        
        d2z/dxx = 2a6 + 2a7y + 2a8y^2
        d2z/dyy = 2a2 + 2a5x + 2a8x^2
        d2z/dxy = a4 + 2a5y + 2a7x + 4a8xy
        """
        # compute the polynomial's hessian
        self.hessx = self.ij[:, 0] - 2
        self.hessx[self.hessx < 0] = 0
        self.hessx = np.vstack([self.hessx, self.ij[:, 1]]).T
        self.hessmaskx = torch.tensor((self.ij[:, 0] > 1) * (self.ij[:, 0]))

        self.hessy = self.ij[:, 1] - 2
        self.hessy[self.hessy < 0] = 0
        self.hessy = np.vstack([self.ij[:, 0], self.hessy]).T
        self.hessmasky = torch.tensor((self.ij[:, 1] > 1) * (self.ij[:, 1]))

        self.hessxy = torch.tensor(self. ij - 1)
        self.hessxy[self.hessxy < 0] = 0
        self.hessmaskxy = self.gradmaskx * self.gradmasky

        # we finally initiate the parameters for which we want to find where the gradient is null
        self.X0 = torch.zeros(2, self.images.shape[-1])
        self.coeffs = vec_polyfit2d(self.x, self.y, self.images, self.ij)

    def forward(self):
        G = polymatrix2d(*self.X0, self.ij)
        return torch.einsum('ij, ji -> i', G, self.coeffs)

    def gradient(self):
        Gx = polymatrix2d(*self.X0, self.gradx)
        Gy = polymatrix2d(*self.X0, self.grady)

        dzdx = torch.einsum('ij, ji -> i', Gx, self.coeffs * self.gradmaskx[:, None])
        dzdy = torch.einsum('ij, ji -> i', Gy, self.coeffs * self.gradmasky[:, None])

        grad = torch.stack((dzdx, dzdy))
        grad = grad.moveaxis(-1, 0)
        return grad

    def hessian(self):
        Gxx = polymatrix2d(*self.X0, self.hessx)
        Gyy = polymatrix2d(*self.X0, self.hessy)
        Gxy = polymatrix2d(*self.X0, self.hessxy)
        d2zdxx = torch.einsum('ij, ji -> i', Gxx, self.coeffs * self.hessmaskx[:, None])
        d2zdyy = torch.einsum('ij, ji -> i', Gyy, self.coeffs * self.hessmasky[:, None])
        d2zdxy = torch.einsum('ij, ji -> i', Gxy, self.coeffs * self.hessmaskxy[:, None])

        hess = torch.stack([torch.stack([d2zdxx, d2zdxy]),
                            torch.stack([d2zdxy, d2zdyy])])
        hess = hess.moveaxis(-1, 0)
        return hess

    def newton(self, niter=20, nugget=1e-6, radius=3):
        reg = torch.eye(2, 2)[None, :]*nugget
        for i in range(niter):
            grad = self.gradient()
            hess = self.hessian()
            inv = torch.linalg.inv(hess + reg)
            dx = torch.einsum('nij, ni -> jn', inv, grad)
            self.X0 = self.X0 - dx

        mask = np.linalg.norm(self.X0, axis=0) > radius
        self.X0[:, mask] = 0

        return self.X0

    def fit(self, n=300, lr=1e-1):
        self.X0 = torch.nn.Parameter(self.X0)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        out = np.nan
        for i in range(n):
            print(f'\riteration {i+1}/{n} = {(i+1) / n * 100:.2f}%, loss = {out:.2f}', end='')
            dx, dy = self.gradient()
            out = torch.sqrt(dx * dx + dy * dy).sum()
            out.backward()
            optimizer.step()
            optimizer.zero_grad()

        return self.X0.detach()

def polyfit2d(x, y, z, order=3, product=False):
    # https://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent

    if product:
        ij = itertools.product(range(order+1), range(order+1))
    else:
        ij = np.vstack((np.arange(order + 1), np.zeros(order + 1)))
        ij = np.hstack((ij, np.roll(ij[:, 1:], 1, axis=0))).T

    ncols = (order + 1)**2 if product else len(ij)
    G = np.zeros((x.size, ncols))

    for k, (i,j) in enumerate(ij):
        G[:, k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z, rcond=1e-10)
    return m


def polyval2d(x, y, m, order, product=True):
    # https://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent
    if product:
        ij = itertools.product(range(order+1), range(order+1))
    else:
        ij = np.vstack((np.arange(order + 1), np.zeros(order + 1)))
        ij = np.hstack((ij, np.roll(ij[:, 1:], 1, axis=0))).T

    z = np.zeros_like(x)
    for a, (i, j) in zip(m, ij):
        z = z + a * x**i * y**j
    return z


def poly_interp(corr, I, J, threshold=1):
    height, width = corr.shape
    h, w = np.arange(height), np.arange(width)
    xx, yy = np.meshgrid(h, w)

    ii = (np.abs(xx - J) <= threshold) & (np.abs(yy - I) <= threshold)

    m = polyfit2d(xx[ii].flatten(), yy[ii].flatten(), corr[ii].flatten(), order=2, product=False)
    Ihat = -0.5 * m[1] / m[2]
    Jhat = -0.5 * m[3] / m[4]

    return Jhat, Ihat


def poly_interp_newton(corr, I, J, degree=2, threshold=1):

    # we define our polynomial f as:
    # f(x, y)=m[0]+m[1]*y+m[2]*yy+m[3]*x+m[4]*x*y+m[5]*x*yy+m[6]*xx+m[7]*xx*y+m[8]*xx*yy
    # df(x, y)/dx = m[3] + m[4] * y + m[5] * yy + m[6] * 2x + m[7] * 2x * y + m[8] * 2x * yy
    # df(x, y)/dy = m[1] + m[2] * 2y + m[4] * x + m[5] * x * 2y + m[7] * xx + m[8] * xx * 2y

    height, width = corr.shape
    h, w = np.arange(height), np.arange(width)
    xx, yy = np.meshgrid(h, w)

    ii = (np.abs(xx - J) <= threshold) & (np.abs(yy - I) <= threshold)
    m = polyfit2d(xx[ii].flatten(), yy[ii].flatten(), corr[ii].flatten(), order=degree, product=True)

    dfdx = lambda X: m[3] + m[4] * X[1] + m[5] * X[1] * X[1] + m[6] * 2 * X[0] + \
                     m[7] * 2 * X[0] * X[1] + m[8] * 2 * X[0] * X[1] * X[1]
    dfdy = lambda X: m[1] + m[2] * 2*X[1] + m[4] * X[0] + m[5] * X[0] * 2 * X[1] + \
                     m[7] * X[0] * X[0] + m[8] * X[0] * X[0] * 2 * X[1]

    grad = lambda X: np.array([dfdx(X), dfdy(X)])

    j0, i0 = root(grad, x0=np.array([J, I])).x

    return i0, j0, m


def poly_interp_library(corr, I, J, degree=2, threshold=2):
    height, width = corr.shape
    h, w = np.arange(height), np.arange(width)
    xx, yy = np.meshgrid(h, w)

    ii = (np.abs(xx - J) <= threshold) & (np.abs(yy - I) <= threshold)
    poly = ndPolynomial(degree=degree)
    X = np.array((xx[ii], yy[ii]))
    poly.fit(X, corr[ii])

    j0, i0 = poly.optimum([J, I])

    return i0, j0, poly.m


def poly_surface(m, I, J, degree=2, threshold=1, N=25):
    x = np.linspace(J - threshold - 0.5, J + threshold + 0.5, N)
    y = np.linspace(I - threshold - 0.5, I + threshold + 0.5, N)
    xx, yy = np.meshgrid(x, y)
    z = polyval2d(xx.flatten(), yy.flatten(), m, order=degree).reshape(xx.shape)
    return xx, yy, z


class ndPolynomial:

    # an attempt at vectorized evaluation of a nd polynomial and it's 1st derivative

    def __init__(self, degree, m=None, powers=None, build_grad=True):
        self.degree = degree
        self.m = m
        self.nd = None
        self.nm = None

        self.grad = None
        self.powers = powers

        # manual lock to not recursively build the gradient?
        self.build_grad = build_grad

    def fit(self, X, z):

        X = self._assert_nd(X)

        self.nd, ns = X.shape
        self.nm = (self.degree + 1) ** self.nd

        G = self._G(X)
        self.m, _, _, _ = np.linalg.lstsq(G, z, rcond=1e-10)

        if self.build_grad:
            self._build_grad()

    def optimum(self, x0):

        x0 = self._assert_nd(x0)

        from scipy.optimize import root
        x0 = root(self.grad, x0=x0, args={'optimum': True}).x
        return x0

    def _assert_nd(self, X):

        if np.isscalar(X):
            X = np.array([X])

        if isinstance(X, list):
            if len(X) == self.nd:
                X = np.array(X)[:, None]
            else:
                raise ValueError('Dimensions of X are inappropriate. X should be of shape (n_dimension, n_obs)')

        if len(X.shape) == 1:
            if self.nd is not None:
                if self.nd == 1:
                    X = np.array(X)[None, :]
                else:
                    assert len(X) == self.nd, 'If inputting a list, length of X should be the same as the number of dimensions'
                    X = np.array(X)[:, None]

        return X.astype(float)

    def _build_grad(self):

        polynomials = []
        ijk = self._ijk()
        for i in range(self.nd):
            derivative = ijk.copy()
            derivative[:, i] -= 1
            derivative[derivative < 0] = 0  # we ensure that we are not raising zero to negative powers
            m = self.m * ijk[:, i]

            poly = ndPolynomial(degree=self.degree-1,
                                m=m,
                                powers=derivative,
                                build_grad=False)
            poly.nd = self.nd
            poly.nm = self.nm
            polynomials.append(poly)

        def grad(X, optimum=False):
            out = np.array([poly(X) for poly in polynomials])

            if optimum:
                out = out[:, 0]


            return out

        self.grad = grad

    def __call__(self, X):
        # we can simply build a corresponding G matrix and multiply it by our coefficient vectors
        G = self._G(X)
        return G @ self.m

    def _ijk(self):
        if self.powers is None:
            ijk = itertools.product(*[range(self.degree + 1) for _ in range(self.nd)])
            ijk = np.array([i for i in ijk])
            return ijk
        return self.powers

    def _G(self, X):
        X = self._assert_nd(X)

        _, ns = X.shape
        ijk = self._ijk()

        #  vectorized construction of the G matrix used for polynomial evaluation
        G = np.stack([X for _ in range(self.nm)])
        G = np.prod(G ** ijk[:, :, None], axis=1)
        return G.T


if __name__ == '__main__':

    x = torch.arange(0, 3) - 1
    X = torch.stack(torch.meshgrid(x, x)).reshape(2, -1)
    z1 = torch.tensor([0.2508, 0.3622, 0.3913, 0.3140, 0.3917, 0.2831, 0.3430, 0.3859, 0.1595])[:, None]
    z2 = torch.tensor([0.4912, 0.5808, 0.1008, 0.4417, 0.5859, 0.1033, 0.3942, 0.5815, 0.0899])[:, None]
    z = torch.hstack((z1, z2))
    poly = PytorchPolyFit2D(X, z2)
    dw, dh = -poly.newton()
    plt.imshow(z2.reshape(3, 3).T, extent=[-1.5, 1.5, -1.5, 1.5])
    plt.scatter(dw, dh)
    xx, yy, zz = poly_surface(poly.coeffs[:, None].numpy(), 0, 0, N=100, threshold=1)
    plt.contourf(xx, yy, zz, alpha=0.3)  # , colors='k', )

    for i in range(len(z)):
        plt.text(*X[:, i].T, f'{z2.numpy()[i, 0]:.5f}')

    x = np.linspace(-1000, 1000, 25)
    z = 5 + 2*x + 0.1 * x * x

    """# 1d example
    poly = ndPolynomial(degree=2)
    poly.fit(x, z)
    zhat = poly(x)
    poly.optimum(10)

    plt.figure()
    plt.scatter(x, z)
    plt.plot(x, zhat)
    plt.title('1d example')"""

    # 2d example
    poly2d = ndPolynomial(degree=2)
    xx, yy = np.meshgrid(x, x)
    X = np.array((xx.flatten(), yy.flatten()))
    Z = 1 + 0.1*X[0] + 0.3*X[0]**2 + 0.6*X[1] + 0.2*X[1]**2 + 0.1*X[0]*X[1] + \
        0.05*X[1]*(X[0]**2) + 0.08*X[0]*(X[1]**2) + 0.01*(X[0]**2)*(X[1]**2)

    poly2d.fit(X, Z)
    Zhat = poly2d(X)

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    #ax.scatter(X[0], X[1], Z)
    x0 = poly2d.optimum([10, 10])
    ax.scatter(*x0, poly2d(x0))
    print(x0)
    ax.plot_surface(xx, yy, Zhat.reshape(xx.shape), alpha=0.5)
    ax.set_title('2d example')

    plt.show()
