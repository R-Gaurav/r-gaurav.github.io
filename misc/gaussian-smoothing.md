This blog details out various methods to implement 1D Gaussian Smoothing in python using numpy library. It would be most useful for the ones who have a basic understanding of Gaussian Smoothing and wish to jump straight to its 
implementation details in python.

With respect to theory, I have linked proper resources througout this blog from where I learned Gaussian Smoothing and gathered
these details. Please feel free to stop by those in case you need more of theoretical knowledge.

#### Prerequisites
  - You understand what smoothing is (not necessarily Gaussian Smoothing).
  - You are familiar with python and numpy library.

In case you do not fulfil the prerequisites, please go through [Smoothing](
http://web.uconn.edu/cunningham/econ397/smoothing.pdf) and [Python's numpy](http://cs231n.github.io/python-numpy-tutorial/).

# Gaussian Smoothing: A brief introduction
In Gaussian Smoothing we use a gaussian kernel to assign weights to the values of an N-dimensional matrix to smooth it. More
information can be found [here](https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm). But, why do we do it? When we gather 
data from real world processes they are often noisy. And when it comes to extracting the signal information from it, it is 
desirable that we get rid of the noise or at least attenuate it by distributing its effect across neighbouring pixels or 
voxels. Through Gaussian Smoothing of data we achieve the same. It also leads to increase in Signal to Noise Ratio (SNR) due to 
[Matched Filtering](https://crewes.org/ForOurSponsors/ResearchReports/2002/2002-46.pdf) process.

### Gaussian Kernel
A 1D Gaussian kernel is mathematically defined as below:

![](/assets/gauss1dEqn.svg)

where \\(\mu\\) is the mean at which the Gaussian kernel is centered and \((\sigma\)) is the standard deviation of this Gaussian distribution. Let us generate and plot an example Gaussian curve with value of \\(\mu\\) = 0 and \\(\sigma\\) = 1.5.  

```
  def _get_1D_gaussian_kernel(self, x_vec, mu=0.0, sigma=1.5):
    """
    Args:
      x_vec (numpy.ndarray): A 1D numpy array.
      mu (float): Mean value around with gaussian function is centered.
      sigma (float): Standard deviation of the gaussian distribution.

    Returns:
      numpy.ndarray: A 1D discrete gaussian kernel array.
    """
    gauss1d = np.exp((-(x_vec - mu)**2) / (2 * sigma**2))
    return gauss1d / np.sum(gauss1d)
```
Note that I have not calculated the coefficient of Gaussian function, because either way it would have been cancelled during the normalization step `gauss1d / np.sum(gauss1d)`. But why normalize? If you recollect, the Gaussian distribution is ideally defined over an infinite interval (\\(-\infty, +\infty\\)), such that the area under the curve is 1.0. Here we have a fixed discrete interval, hence normalization by the sum of all values in kernel ensures that all the values in the kernel add upto 1.0 after normalization. 

Below is the 1D Gaussian kernel with values of x-axis spanning from \[-10, 10\] with a step of 0.2. 

![](/assets/1D-Gaussian-Kernel.png)

## 1D Gaussian Smoothing
Now let us see 1D Gaussian smoothing in action. First I will draw a sine curve and then add a Gaussian distributed noise to it. Next, after smoothing the curve we will validate if Gaussian smoothing has attenuated the noise or not? But you see... there are various ways to do it, and we will explore them one by one.

### Naive 1D Smoothing
This is achieved by calculating the Gaussian smoothed value at every point on the curve. Below code implements this method.
