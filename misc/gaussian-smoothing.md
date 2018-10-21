This blog details out various methods to implement Gaussian Smoothing (1D, 2D and 3D) in python using numpy library. It would
be most useful for the ones who have a basic understanding of Gaussian Smoothing and wish to jump straight to its 
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

### Gaussian Kernel
A 1D Gaussian Kernel is mathematically defined as below:

![](/assets/gauss1dEqn.svg)

where \(\sigma(\sum_iw_ix_i + b)\)
