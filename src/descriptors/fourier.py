import numpy as np
from scipy.fft import fftn
import gc
import scipy.ndimage

class FourierDescriptor:
    def __init__(self, resolution=32):
        self.resolution = resolution

    def compute(self, voxel_grid):
        # Add normalization for translation invariance
        voxel_grid = voxel_grid - np.mean(voxel_grid)

        # Power spectrum for rotation invariance
        fft = fftn(voxel_grid)
        power_spectrum = np.abs(fft) ** 2

        # Shell-based features with improved radius sampling
        features = []
        center = np.array(power_spectrum.shape) // 2
        max_radius = min(center)

        # Log-space radius sampling for better feature distribution
        radii = np.logspace(0, np.log10(max_radius), 32)

        for r1, r2 in zip(radii[:-1], radii[1:]):
            shell_mask = self._get_shell_mask(power_spectrum.shape, center, r1, r2)
            shell_values = power_spectrum[shell_mask]

            if len(shell_values) > 0:
                features.extend([
                    np.mean(shell_values),
                    np.std(shell_values),
                    np.percentile(shell_values, 75)
                ])

        return self._normalize_features(np.array(features))

    def _get_shell_mask(self, shape, center, r1, r2):
        x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
        distances = np.sqrt(
            (x - center[0])**2 +
            (y - center[1])**2 +
            (z - center[2])**2
        )
        return (distances >= r1) & (distances < r2)

    def _downsample(self, array, factor):
        if factor >= 1:
            return array
        return scipy.ndimage.zoom(array, (factor, factor, factor))

    def _pad_volume(self, voxel_grid):
        target_shape = (self.resolution, self.resolution, self.resolution)
        pad_width = [(0, max(0, target - s)) for target, s in zip(target_shape, voxel_grid.shape)]
        return np.pad(voxel_grid, pad_width, mode='constant')

    def _extract_features(self, magnitude_spectrum, max_freq=32):
        features = []
        center = np.array(magnitude_spectrum.shape) // 2
        x, y, z = np.ogrid[:magnitude_spectrum.shape[0],
                  :magnitude_spectrum.shape[1],
                  :magnitude_spectrum.shape[2]]

        for radius in range(1, max_freq):
            # Calculate distances from center
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
            shell_mask = (dist >= radius - 0.5) & (dist < radius + 0.5)
            shell_values = magnitude_spectrum[shell_mask]

            # Handle empty shells
            if len(shell_values) == 0:
                features.extend([0.0, 0.0, 0.0])  # Use zeros for empty shells
            else:
                features.extend([
                    np.mean(shell_values),
                    np.std(shell_values) if len(shell_values) > 1 else 0.0,
                    np.max(shell_values)
                ])

        return np.array(features)

    def _normalize_features(self, features):
        norm = np.linalg.norm(features)
        if norm < 1e-10:
            return np.zeros_like(features)
        return features / norm