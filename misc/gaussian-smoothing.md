This blog details out various methods to implement 1D Gaussian Smoothing in python using numpy library. It would be most useful for the ones who have a basic understanding of Gaussian Smoothing and wish to jump straight to its 
implementation details in python.

With respect to theory, I have linked proper resources througout this blog from where I learned Gaussian Smoothing and gathered
these details. Please feel free to stop by those in case you need more of theoretical knowledge.

#### Prerequisites
  - You understand what smoothing is (not necessarily Gaussian Smoothing).
  - You are familiar with python and numpy library.

In case you do not fulfil the prerequisites, please go through [Smoothing](
http://web.uconn.edu/cunningham/econ397/smoothing.pdf) and [Python's numpy](http://cs231n.github.io/python-numpy-tutorial/).

## Gaussian Smoothing: A brief introduction
In Gaussian Smoothing we use a gaussian kernel to assign weights to the values of an N-dimensional matrix to smooth it. More
information can be found [here](https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm). But, why do we do it? When we gather 
data from real world processes they are often noisy. And when it comes to extracting the signal information from it, it is 
desirable that we get rid of the noise or at least attenuate it by distributing its effect across neighbouring pixels or 
voxels. Through Gaussian Smoothing of data we achieve the same. It also leads to increase in Signal to Noise Ratio (SNR) due to 
[Matched Filtering](https://crewes.org/ForOurSponsors/ResearchReports/2002/2002-46.pdf) process.

## Gaussian Kernel
A 1D Gaussian kernel is mathematically defined as below:

![](/assets/gauss1dEqn.svg)

where \\(\mu\\) is the mean at which the Gaussian kernel is centered and \((\sigma\)) is the standard deviation of this Gaussian distribution. Let us generate and plot an example Gaussian curve with value of \\(\mu\\) = 0 and \\(\sigma\\) = 1.5.  

```
# 1D Gaussian Kernel.
  def _get_1D_gaussian_kernel(self, mu=0.0, sigma=1.5):
    """
    Args:
      mu (float): Mean value around with gaussian function is centered.
      sigma (float): Standard deviation of the gaussian distribution.

    Returns:
      numpy.ndarray: A 1D discrete gaussian kernel array.
    """
    gauss1D = np.exp((-(self._x_vec - mu)**2) / (2 * sigma**2))
    return gauss1D / np.sum(gauss1D)
```
Note that I have not calculated the coefficient of Gaussian function, because either way it would have been cancelled during the normalization step `gauss1d / np.sum(gauss1d)`. But why normalize? If you recollect, the Gaussian distribution is ideally defined over an infinite interval (\\(-\infty, +\infty\\)), such that the area under the curve is 1.0. Here we have a fixed discrete interval, hence normalization by the sum of all values in kernel ensures that all the values in the kernel add upto 1.0 after normalization. 

Below is the 1D Gaussian kernel with values of x-axis spanning from \[-10, 10\] with a step of 0.2 (values in `self._x_vec`). 

![](/assets/1D-Gaussian-Kernel.png)

## 1D Gaussian Smoothing
Now let us see 1D Gaussian smoothing in action. First I will draw a quadratic curve and then add a Gaussian distributed noise to it. Below code does the necessary.
```
# Generating a quadratic curve.
  def _get_1D_quadratic_curve(self):
    """
    Returns:
      numpy.ndarray: A 1D discrete quadratic curve array.
    """
    return self._x_vec ** 2
    
# Generating random noise with amplitude 5 units.
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
    
# Adding noise to the quadratic curve.
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
```
Following curves are obtained as a result:
![](/assets/QCandNQC.png)

Next, let us smooth the noisy curve and validate if Gaussian smoothing has attenuated the noise or not? But you see... there are various ways to do it, and we will explore them one by one.

### Naive 1D Smoothing
This is achieved by calculating the Gaussian smoothed value at every point on the curve. Below code implements this method.
```
# Naive 1D Smoothing.
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
```
As you can see from the above function `def apply_naive_1D_gaussian_smoothing(self, noisy_curve)`, a Gaussian kernel is calculated with mean \\(\mu\\) as each value in `self._x_vec`. If you try to imagine the whole process executed in the function, it would be evident that the process is equivalent to convolution with a symmetric kernel. So there lies our second method to perform smoothing. But before I proceed on it, let me plot the smoothed curve obtained after naive 1D smoothing.

![](/assets/NaiveSC.png)
As can be seen from the above plot, naively smoothed curve is very much closer to the original curve.

### Smoothing by Convolution

