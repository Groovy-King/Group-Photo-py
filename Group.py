import numpy as np
from util import *
from constants import *
import astropy.units as u
from astropy.coordinates import SkyCoord

class Group:
    def __init__(self, galaxy, m500):
        """
            Must inlcude M500 in the initialization, and the galaxy must have a redshift (pos[2]) for the calculations to work.
        """
        self.galaxy = galaxy
        self.pos = galaxy.pos
        self.m500 = m500
        return

    def NFW_cylinder(self, galaxies):
        try:
            H = self.H
        except KeyError:
            self.compute_properties()
            H = self.H
        r200 = self.r200
        sigma_z = self.sigma_z
        Da = self.Da
        theta_max = (r200 / Da).to(u.dimensionless_unscaled).value
        z_max = 2 * sigma_z.value

        # Extract the RA, Dec, and redshift of the group and the input galaxies
        ra_group = self.pos[0]
        dec_group = self.pos[1]
        z_group = self.pos[2]
        ra_galaxies = np.array([g.pos[0] for g in galaxies], dtype = object)
        dec_galaxies = np.array([g.pos[1] for g in galaxies], dtype = object)
        z_galaxies = np.array([g.pos[2] for g in galaxies])

        # Compute the projected radius R for each galaxy in the input list
        sky_group = SkyCoord(ra = ra_group, dec = dec_group)
        sky_galaxies = SkyCoord(ra = ra_galaxies, dec = dec_galaxies)
        angular_separation = sky_group.separation(sky_galaxies).to(u.rad, equivalencies = u.dimensionless_angles()).value
        angular_mask = angular_separation < theta_max

        # Compute the line-of-sight velocity difference for each galaxy
        delta_z = z_galaxies - z_group
        z_mask = np.abs(delta_z) < z_max

        # Combine the angular and redshift masks to identify galaxies within the NFW cylinder
        nfw_cylinder_mask = angular_mask & z_mask
        return nfw_cylinder_mask
    
    def _f(self, x):
        """Computes the function f(x) used in the NFW profile for the projected surface density. Refer to equation 15 in the draft paper."""
        if x < 0:
            raise ValueError("x must be non-negative!")
        elif x < 1:
            return 1 - 2*np.arctanh( np.sqrt((1 - x) / (1 + x)) ) / np.sqrt(1 - x**2)
        elif x == 1:
            return 0
        else:
            return 1 - 2*np.arctan( np.sqrt((x - 1) / (x + 1)) ) / np.sqrt(x**2 - 1)
    
    def _NFW_Sigma(self, r):
        """Computes the projected surface density Sigma(r) at a given projected radius r using the NFW profile. Refer to equation 14 in the draft paper."""
        try:
            r200 = self.r200
            c200 = self.c200
            rho_crit = self.rho_crit
        except KeyError:
            self.compute_properties()
            r200 = self.r200
            c200 = self.c200
            rho_crit = self.rho_crit

        # Compute the scale radius rs
        rs = r200 / c200

        # Compute the characteristic density rho_s
        rho_s = 200 * rho_crit * (c200**3) / (3 * (np.log(1 + c200) - c200 / (1 + c200)))

        # Compute the projected surface density Sigma(r) using the NFW profile
        x = r / rs
        x = x.to(u.dimensionless_unscaled).value  # Convert x to a dimensionless value for the function f(x)
        Sigma = 2 * rho_s * rs * self._f(x) / (x**2 - 1)
        return Sigma
    
    def compute_properties(self):
        """
            Computes R200, M200, c200, and rho_crit for the group based on the input M500 and redshift.
        """
        m500 = self.m500
        z = self.pos[2]

        # Compute the critical density of the universe at the galaxy's redshift
        H = Hubble_parameter(z)
        self.H = H

        Da = angular_diameter_distance(z)
        self.Da = Da.to(u.Mpc)

        rho_crit = 3 * H**2 / (8 * np.pi * G) 
        self.rho_crit = rho_crit.to(u.Msun / u.Mpc**3)

        # Compute the radius R500 corresponding to M500
        r500 = (3 * m500 / (4 * np.pi * 500 * rho_crit))**(1/3)
        r500 = r500.to(u.Mpc)
        self.r500 = r500

        # Compute M200 using the relation r200 = r500 / 0.67
        r200 = r500 / 0.67
        self.r200 = r200.to(u.Mpc)

        m200 = (4/3) * np.pi * 200 * rho_crit * r200**3
        m200 = m200.to(u.Msun)
        self.m200 = m200

        # Compute the concentration parameter c200 using the Biviano et al. (2017) relation, check equation 17 of draft paper
        c200 = 10**(1.9 - 0.1 * np.log10(m200.to(u.Msun).value))
        self.c200 = c200

        sigma_e = 0.26
        sigma_e2 = sigma_e * np.log(10) / 3
        E = H / H0
        A = (474.4 * u.km / u.s)**3 * (m500 * E / (1e14 * u.Msun))**(0.94)
        A = A.to(u.km**3 / u.s**3)
        sigma_v = A**(1/3) * np.exp(sigma_e2**2 / 2)
        self.sigma_v = sigma_v.to(u.km / u.s)

        sigma_z =  (1 + z) * sigma_v / c
        self.sigma_z = sigma_z.to(u.dimensionless_unscaled)
        return
