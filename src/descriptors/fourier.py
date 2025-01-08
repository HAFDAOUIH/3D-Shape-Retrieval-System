import numpy as np
from scipy.fft import fftn
import gc
import scipy.ndimage

class FourierDescriptor:
    def __init__(self, resolution=32):
        self.resolution = resolution

    def compute(self, voxel_grid):
        try:
            # Ensure input is the right size
            if voxel_grid.shape[0] > self.resolution:
                factor = self.resolution / voxel_grid.shape[0]
                voxel_grid = self._downsample(voxel_grid, factor)

            # Pad the volume
            padded = self._pad_volume(voxel_grid)

            # Compute FFT
            fft = fftn(padded)
            magnitude_spectrum = np.abs(fft)

            # Extract features
            features = self._extract_features(magnitude_spectrum)

            # Cleanup
            del fft, magnitude_spectrum
            gc.collect()

            return self._normalize_features(features)
        except Exception as e:
            print(f"Error in Fourier descriptor computation: {str(e)}")
            raise

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