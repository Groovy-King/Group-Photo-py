import numpy as np
import astropy.units as u
from scipy.integrate import quad

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