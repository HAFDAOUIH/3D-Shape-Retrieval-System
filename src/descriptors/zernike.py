import numpy as np
from scipy.special import sph_harm, factorial
import gc
import scipy.ndimage

class ZernikeDescriptor:
    def __init__(self, max_order=10):
        self.max_order = max_order
        self.resolution = 32

    def compute(self, voxel_grid):
        if voxel_grid.shape[0] > self.resolution:
            factor = self.resolution / voxel_grid.shape[0]
            voxel_grid = self._downsample(voxel_grid, factor)

        r, theta, phi = self._to_spherical_coords(voxel_grid)
        moments = []

        for n in range(0, self.max_order + 1, 2):
            for l in range(n + 1):
                if (n - l) % 2 == 0:
                    for m in range(-l, l + 1):
                        moment = self._compute_moment(voxel_grid, r, theta, phi, n, l, m)
                        moments.append(abs(moment))

        return self._normalize_moments(np.array(moments))

    def _downsample(self, array, factor):
        if factor >= 1:
            return array
        return scipy.ndimage.zoom(array, factor)

    def _to_spherical_coords(self, voxel_grid):
        grid_size = voxel_grid.shape[0]
        x, y, z = np.meshgrid(
            np.linspace(-1, 1, grid_size),
            np.linspace(-1, 1, grid_size),
            np.linspace(-1, 1, grid_size),
            indexing='ij'
        )

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z/(r + 1e-10))
        phi = np.arctan2(y, x)

        return r, theta, phi

    def _compute_moment(self, voxel_grid, r, theta, phi, n, l, m):
        R = self._radial_polynomial(n, l, r)
        Y = sph_harm(m, l, phi, theta)
        integrand = voxel_grid * R * np.real(Y)
        return np.sum(integrand)

    def _radial_polynomial(self, n, l, r):
        R = np.zeros_like(r)
        for k in range((n-l)//2 + 1):
            coef = (-1)**k * factorial(n-k)
            coef /= factorial(k) * factorial((n+l)//2 - k) * factorial((n-l)//2 - k)
            R += coef * r**(n-2*k)
        return R

    def _normalize_moments(self, moments):
        return moments / (np.linalg.norm(moments) + 1e-10)
