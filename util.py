import numpy as np
import astropy.units as u
from scipy.integrate import quad
from scipy.special import gammaincc

from constants import *

# Compute the Hubble constant at a given redshift
def Hubble_parameter(z):
    return H0 * np.sqrt(omg_m * (1 + z)**3 + omg_lambda + omg_k * (1 + z)**2)

# Compute the comoving distance to a given redshift
def comoving_distance(z):
    integrand = lambda z_prime: (c/ Hubble_parameter(z_prime)).to(u.Mpc).value
    distance, _ = quad(integrand, 0, z)
    return distance * u.Mpc

# Compute the angular diameter distance to a given redshift
def angular_diameter_distance(z):
    d_c = comoving_distance(z)
    return d_c / (1 + z)

def schechter_luminosity_function(M, M_star = (-21.35 + 5*np.log10(h)) * u.mag, Phi_star = 1.49e-2 * h**3 * u.Mpc**-3 / u.mag, alpha = -1.3, return_units = True):
    """
    Computes the Schechter luminosity function for a given absolute magnitude M, with parameters M_star, Phi_star, and alpha. 
    The function returns the number density of galaxies per unit magnitude at the specified absolute magnitude.
    """
    L = 10**(0.4 * (M_star.value - M.value))  # Convert absolute magnitude to luminosity relative to L_star
    phi = 0.4 * (Phi_star * np.log(10)) * (L**(alpha + 1)) * np.exp(-L)  # Schechter function formula
    phi = phi.to(u.Mpc**-3 / u.mag)
    if return_units:
        return phi  # Return the result with appropriate units
    else:
        return phi.value  # Return the numerical value without units

def cumulative_luminosity_function(M, M_star = (-21.35 + 5*np.log10(h)) * u.mag, Phi_star = 1.49e-2 * h**3 * u.Mpc**-3 / u.mag, alpha = -1.3, return_units = True):
    """
    Computes the cumulative luminosity function, which gives the number density of galaxies brighter than a given absolute magnitude M. 
    This is obtained by integrating the Schechter luminosity function from -infinity to M.
    """
    x = 10**(0.4 * (M_star.value - M.value))  # Convert absolute magnitude to luminosity relative to L_star
    cumulative_density = Phi_star * gammaincc(alpha + 1, x)  # Analytical expression for the cumulative luminosity function using the regularized incomplete gamma function
    cumulative_density = cumulative_density.to(u.Mpc**-3) * u.mag  # Convert to number density units
    if return_units:
        return cumulative_density  # Return the result with appropriate units
    else:
        return cumulative_density.value  # Return the numerical value without units