#!/usr/bin/env python

import numpy as np
from scipy.special import j1
from scipy.integrate import cumtrapz
from scipy.stats import norm
from numpy.polynomial.polynomial import polyval2d
from time import time

##############################################################################
###    The following functions generate photons
##############################################################################

# Indexing constants for photon arrays
# P is used as a probability of detection
X, Y, P = 0, 1, 2 

def airy_disk(x=0, y=0, ang_size=1., N=2**18):
    """
    Uses Inverse Transform Sampling to generate 
    an airy disk pattern of desired size. Angular size is
    defined here by the first null of the bessel function.
    Inputs:
        x        - x-coord of origin in arcsec
        y        - y-coord of origin in arcsec
        ang_size - angular size in arcsec
        N        - number of photons to generate
    Outputs:
        photons  - an array of photons with shape=(3,N)
            the first 2 dimensions are spatial locations,
            the last contains the photon's detection prob.
    """
    # compute cdf numerically, going out to tenth minima 
    airy = lambda m: 3*np.pi/8*(j1(2*m)/m)**2  
    ms   = np.linspace(1e-12,16.1,100000) 
    cdf  = cumtrapz(airy(ms),ms,initial=0)
    
    # draw samples via inverse transform sampling and convert to angular size
    cdf_samps = np.random.uniform(0,cdf[-1],N)
    rs = ms[np.searchsorted(cdf,cdf_samps)]/1.916*ang_size/2

    # convert to cartesian coords and set detection probabilities to 1
    thetas  = np.random.uniform(0,2*np.pi,N)
    photons = np.empty((3,N),np.float32)
    photons[[X,Y]] = x+rs*np.cos(thetas), y+rs*np.sin(thetas)
    photons[P] = 1.
    return photons


def gaussian2D(x=0, y=0, cov=[[1,0],[0,1]], N=2**18):
    """
    2D Gaussian profile.
    Inputs:
        x   - x-coord of origin in arcsec
        y   - y-coord of origin in arcsec
        cov - covariance matrix used to define profile shape
        N   - number of photons to generate
    Outputs:
        photons  - an array of photons with shape=(3,N)
            the first 2 dimensions are spatial locations,
            the last contains the photon's detection prob.
    """
    photons = np.empty((3,N))
    photons[[X,Y]] = np.random.multivariate_normal([x,y],cov,N).T
    photons[P] = 1.
    return photons


def sky_background(pixel_size, ccd_dim, N):
    """
    Uniform random noise.
    Inputs:
        pixel_size - angular size of pixels in arcsec
        ccd_dim    - Scalar if detector is square, tuple/list otherwise
                     (ex. ccd_dim == 16 gives a 16x16 CCD)
        N          - number of photons to generate
    Outputs:
        photons  - an array of photons with shape=(3,N)
            the first 2 dimensions are spatial locations,
            the last contains the photon's detection prob.
    """
    photons = np.ones((3,N),np.float32)
    if np.isscalar(ccd_dim):
        photons[[X,Y]] = np.random.uniform(-ccd_dim/2,ccd_dim/2,(2,N))*pixel_size
    else:
        photons[X] = np.random.uniform(-ccd_dim[X]/2,ccd_dim[X]/2,N)*pixel_size
        photons[Y] = np.random.uniform(-ccd_dim[Y]/2,ccd_dim[Y]/2,N)*pixel_size
    return photons


def cosmic_ray(x0, x1, y0, y1, N):
    """
    Streak caused by cosmic rays.
    Inputs:
        x0, x1, y0, y1 - starting/ending coords in arcsec
        N              - number of photons to generate
    Outputs:
        photons  - an array of photons with shape=(3,N)
            the first 2 dimensions are spatial locations,
            the last contains the photon's detection prob.
    """
    photons = np.ones((3,N),np.float32)
    photons[[X,Y]] = np.linspace(x0,x1,N), np.linspace(y0,y1,N)
    return photons



##############################################################################
###    The following functions simulate detector sensitivity 
###    by modifying the detection probability of photons
##############################################################################

def pixel_coords(photons, pixel_size=0.073):
    """ 
    Remap coordinates of photons to per-pixel coordinates.
    These coordinates run from -0.5 to 0.5 in both dimensions.
    Inputs:
        photons    - Array of photons with shape=(3,N_photons)
        pixel_size - Angular size of pixels in arcsec
    Outputs:
        pix        - Pixel coordinates with shape=(2,N_photons)
    """
    pix = np.modf(photons[[X,Y]]/pixel_size)[0]
    return np.where(pix < 0, pix+0.5, pix-0.5)


def apply_pixel_gap(photons, pixel_size=0.073, gap=0.00073):
    """
    Mask out the gaps between pixels.
    Inputs:
        photons    - Array of photons with shape=(3,N_photons)
        pixel_size - Angular size of pixels in arcsec 
        gap        - Angular size of gap between pixels in arcsec
    Outputs:
        None, photons[P] (probabilities of detection) is modified in place 
    """
    assert gap < pixel_size, "Gaps are bigger than pixel!"
    gap /= pixel_size  # gap as a fractional size of pixel
    gap /= 2           # each pixel has a border of size gap/2
    pix = pixel_coords(photons, pixel_size)
    photons[P,np.any(0.5-np.abs(pix) < gap,0)] = 0 
        

def apply_intra_pixel(photons, pixel_size=0.073, scale=0.7):
    """ 
    Gaussian sensitivity variations that peak in pixel centers
    Inputs:
        telescope - Telescope object to remap photons to pixel coords
        photons   - Array of photons with shape=(3,N_photons)
        scale     - Scale for 2D Gaussian w.r.t. pixel size
    Outputs:
        None, photons[P] (probabilities of detection) is modified in place
    """ 
    pix = pixel_coords(photons, pixel_size)
    d = np.sqrt((pix**2).sum(0)) # distances from pixel centers
    photons[P] *= norm.pdf(d,scale=scale)/norm.pdf(0,scale=scale)


def apply_inter_pixel(photons, coef, scale=0.2):
    """ 
    2D polynomial sensitivity variations.
    scale = 0.2 will produce a sensitivity range of 80-100%  
    Inputs: 
        photons - Array of photons
        coef    - Coefficient matrix
        scale   - Either 1) Percent of desired variations in range [0, 1]
                         2) Previously returned scale list
    Outputs: 
        scale   - Scaling used 
        **photons[P] (probabilities of detection) is also modified in place**
    """
    poly  = polyval2d(photons[X],photons[Y],coef)
    if np.isscalar(scale):
        assert scale <= 1 and scale >= 0, "scale must be in [0,1]"
        scale = [poly.min(),scale/(poly.max()-poly.min()),1-scale]
    poly -= scale[0]
    poly *= scale[1]
    poly += scale[2]
    photons[P] *= poly
    return scale


def apply_quantum_efficiency(photons, QE=0.9):
    """ 
    Quantum efficiency for photon detections.
    Inputs:
        photons - Array of photons with shape=(3,N_photons)
        QE      - Quantum efficiency factor
    Outputs:
        None, photons[P] (probabilities of detection) is modified in place
    """
    photons[P] *= QE



##############################################################################
###    Misc. Detector functions
##############################################################################

def bin_photons(photons, pixel_size, ccd_dim, subgrid=1.):
    """ 
    Bin photons according to detector shape. Photons are filtered by their
    probability of detection, which is modified by above sensitivity functions
    Inputs:
        photons    - Array of photons with shape=(3,N_photons)
        pixel_size - Angular size of pixels in arcsec
        ccd_dim    - Scalar if detector is square, tuple/list otherwise
                     (ex. ccd_dim == 16 gives a 16x16 CCD)
        subgrid    - Factor to subdivide pixels by (default = 1)
    Outputs:
        binned (via np.histogram2d) matrix giving counts per pixel
    """
    detected = photons[P] > np.random.rand(photons.shape[1])
    if np.isscalar(ccd_dim):
        bins = np.linspace(-ccd_dim/2,ccd_dim/2,ccd_dim*subgrid+1)*pixel_size
    else:
        bins = [np.linspace(-d/2,d/2,d*subgrid+1)*pixel_size for d in ccd_dim]
    return np.histogram2d(photons[X,detected],photons[Y,detected],bins)[0]


def apply_read_noise(pixels, rms=4):
    """
    Apply read noise to pixelated counts.
    """
    pixels += np.round(np.random.normal(scale=rms,size=pixels.shape))
    pixels[pixels < 0] = 0


def apply_dead_pixel_rows(pixels, n):
    """
    Kills some pixel rows
    """
    pixels[np.random.randint(0,len(pixels),n)] = 0



##############################################################################

def main():
   
    ##### Typical useage example

    pixel_size = 0.073  # Gemini pixel size (arcsec)
    ccd_dim    = 20     # gives 20x20 CCD

    # generate photons
    point_source = airy_disk(ang_size=3*pixel_size, N=np.random.poisson(2**20))
    background   = sky_background(pixel_size, ccd_dim, np.random.poisson(2**16))    
    all_photons  = np.hstack((point_source, background))

    # apply sensitivity variations
    apply_pixel_gap(all_photons, pixel_size, pixel_size/100)
    apply_intra_pixel(all_photons, pixel_size)
    apply_inter_pixel(all_photons, coef=np.random.randn(4,4))
    apply_quantum_efficiency(all_photons)
    
    # bin photons (convolves applied sensitivities)
    pixelated = bin_photons(all_photons, pixel_size, ccd_dim)
    apply_read_noise(pixelated)

    # now you can do stuff like plt.imshow(pixelated)...

    
    ###########################################################################

    ##### Quick plotting example - Airy disk

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    pixel_size = 0.073
    ccd_dim    = 10
    photons    = airy_disk(0,0,2*pixel_size,1000000)
    pixelated  = bin_photons(photons, pixel_size, ccd_dim, subgrid=10)

    L = pixel_size*ccd_dim 
    extent = [-L/2, L/2, -L/2, L/2]
    im = plt.imshow(pixelated,interpolation='none',extent=extent,
                    origin='lower',cmap='cool',norm=LogNorm())
    cb = plt.colorbar(im,pad=0,label='Count')
    plt.xlabel('x [arcsec]',size=16)
    plt.ylabel('y [arcsec]',size=16)
    plt.savefig('airy_disk.png')
    print "Made 'airy_disk.png'"

if __name__=='__main__': main()

