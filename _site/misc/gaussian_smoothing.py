#
# This file implements gaussian smoothing.
#

import matplotlib.pyplot as plt
import numpy as np

class GaussianSmoothing(object):
  def __init__(self, limit, step):
    self._x_vec = np.arange(-limit, limit+step, step)

  def _get_1D_gaussian_kernel(self, mu=0.0, sigma=1.5):
    """
    Args:
      x_vec (numpy.ndarray): A 1D numpy array.
      mu (float): Mean value around with gaussian function is centered.
      sigma (float): Standard deviation of the gaussian distribution.

    Returns:
      numpy.ndarray: A 1D discrete gaussian kernel array.
    """
    gauss1D = np.exp((-(self._x_vec - mu)**2) / (2 * sigma**2))
    return gauss1D / np.sum(gauss1D)

  def _get_1D_quadratic_curve(self):
    """
    Args:
      x_vec (numpy.ndarray): A 1D numpy array.

    Returns:
      numpy.ndarray: A 1D discrete quadratic curve array.
    """
    return self._x_vec ** 2

  def _get_normal_noise(self, size, mu=0, sigma=1):
    """
    Args:
      size (int): Size of the noise array.
      mu (float): Mean of normal (gaussian) distribution.
      sigma (float): Standard deviation of the normal distribution.

    Returns:
      numpy.ndarray: A 1D noise array with number of elements equal to size.
    """
    return 5*np.random.normal(mu, sigma, size)

  def _get_noisy_1D_quadratic_curve(self):
    """
    Args:
      x_vec (numpy.ndarray): A 1D numpy array.

    Returns:
      numpy.ndarray: A 1D noisy quadratic curve.
    """
    quad_curve = self._get_1D_quadratic_curve()
    size = x_vec.shape[0]
    noise = self._get_normal_noise(size)
    return quad_curve + noise

  def apply_naive_1D_gaussian_smoothing(self, noisy_curve):
    """
    Args:
      noisy_curve (numpy.ndarray): A 1D noisy curve.

    Returns:
      numpy.ndarray: A 1D smoothed curve.
    """
    len_nc = noisy_curve.shape[0]
    smoothed_array = []

    for i in range(len_nc):
      gauss1D = self._get_1D_gaussian_kernel(mu=self._x_vec[i])
      smoothed_array.append(np.sum(gauss1D * noisy_curve))

    return np.array(smoothed_array)

if __name__ == "__main__":
  gs = GaussianSmoothing(10, 0.2)
  # Get a 1D Gaussian Kernel.
  x_vec = np.arange(-10, 10.2, 0.2)
  quad_curve = gs._get_1D_quadratic_curve()
  noisy_quad_curve = gs._get_noisy_1D_quadratic_curve()
  smoothed_array = gs.apply_naive_1D_gaussian_smoothing(noisy_quad_curve)

  plt.plot(x_vec, quad_curve, label="Quadratic Curve")
  plt.plot(x_vec, noisy_quad_curve, label="Noisy Quadratic Curve")
  plt.plot(x_vec, smoothed_array, label="Smoothed Curve")

  plt.legend(loc="upper right")
  plt.show()
