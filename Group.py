import numpy as np
import os
from util import *
from constants import *
import astropy.units as u
from astropy.coordinates import SkyCoord
from collections.abc import Iterable
from Member_galaxy import MemberGalaxy
from Galaxy import Galaxy
from statsmodels.nonparametric.smoothers_lowess import lowess

class Group:
    # Generate a unique ID for each group instance, while storing the count as a class variable to ensure uniqueness across all instances
    _id_counter = 0
    @classmethod
    def _generate_id(cls):
        pid = os.getpid()  # Get the current process ID to ensure uniqueness across different runs
        id = f"{pid}_{cls._id_counter}"  # Combine process ID with the ID counter
        cls._id_counter += 1
        return id
    
    # Define equality and hashing methods to allow Group instances to be stored in sets and compared based on their unique IDs
    def __eq__(self, value):
        return self.id == value.id
    
    def __hash__(self):
        return hash(self.id)
    
    # Constructor to initialize a Group instance with its central galaxy and M500. The position of the group is set to be the same 
    # as the central galaxy's position. An empty set is initialized to store the galaxies that belong to this group.
    def __init__(self, central_galaxy, m500):
        """
            Must inlcude M500 in the initialization, and must always be called within a Galaxy instance.
        """
        self.central_galaxy = central_galaxy
        self.pos = central_galaxy.pos
        self.m500 = m500
        self.id = self._generate_id()  # Unique identifier for the group, generated using a combination of the central galaxy's ID and the mass band index
        self._member_galaxies = set()  # Initialize an empty set to store the galaxies that belong to this group
        self.add_galaxy(central_galaxy)  # Add the central galaxy to the group by default
        return
    
    def get_member_galaxies(self):
        """
            Returns the set of member galaxies that belong to this group. 
            This method can be used to access the galaxies that are part of the group after they have been added using the add_galaxies method.
        """
        return self._member_galaxies

    def NFW_cylinder(self, angular_separation, delta_z):
        try:
            H = self.H
        except AttributeError:
            self.compute_properties()
            H = self.H
        r200 = self.r200
        sigma_z = self.sigma_z
        Da = self.Da
        theta_max = (r200 / Da).to(u.dimensionless_unscaled).value
        z_max = 2 * sigma_z.value
        
        # Generate a boolean mask for galaxies within the angular separation corresponding to R200
        angular_mask = angular_separation < theta_max

        # Generate a boolean mask for galaxies within the redshift separation corresponding to 2*sigma_z
        z_mask = np.abs(delta_z) < z_max

        # Combine the angular and redshift masks to identify galaxies within the NFW cylinder
        nfw_cylinder_mask = angular_mask & z_mask

        # Store the richness of the group as the number of galaxies within the NFW cylinder
        self.richness = np.sum(nfw_cylinder_mask)

        return nfw_cylinder_mask
    
    def _f(self, x):
        """Computes the function f(x) used in the NFW profile for the projected surface density. Refer to equation 15 in the draft paper."""
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x)
        eps = 1e-8

        # x < 1
        m1 = x < 1 - eps
        xm = x[m1]
        result[m1] = (1 - np.arccosh(1/xm) / np.sqrt(1 - xm**2)) / (xm**2 - 1)

        # x ≈ 1 (analytic limit)
        m2 = np.abs(x - 1) <= eps
        result[m2] = 1/3  # IMPORTANT: finite limit for this form

        # x > 1
        m3 = x > 1 + eps
        xp = x[m3]
        result[m3] = (1 - np.arccos(1/xp) / np.sqrt(xp**2 - 1)) / (xp**2 - 1)

        return result
    
    def NFW_Sigma(self, r):
        """
        Computes the projected surface density Sigma(r) at a given projected radius r using the NFW profile. 
        Refer to equation 14 in the draft paper.
        New version uses Eqn. 41 from Lokas and Mamon (2001), avoiding the issue of singularity at r = rs. 
        Old version used Eqn. 2.6 and 2.7 from Bartelmann (1996).
        """
        try:
            r200 = self.r200
            c200 = self.c200
            rho_crit = self.rho_crit
        except AttributeError:
            self.compute_properties()
            r200 = self.r200
            c200 = self.c200
            rho_crit = self.rho_crit

        # Compute the scale radius rs
        rs = r200 / c200
        self.rs = rs.to(u.Mpc)

        # Compute the characteristic density rho_s
        rho_s = 200 * rho_crit * (c200**3) / (3 * (np.log(1 + c200) - c200 / (1 + c200)))
        self.rho_s = rho_s

        # Compute the projected surface density Sigma(r) using the NFW profile
        x = r / rs
        x = x.to(u.dimensionless_unscaled).value  # Convert x to a dimensionless value for the function f(x)
        Sigma = 2 * rho_s * rs * self._f(x)
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

        # Compute the velocity dispersion sigma_v using the relation from Pearson et al. (2012), check equation 10 of draft paper
        sigma_v = A**(1/3) * np.exp(sigma_e2**2 / 2)
        self.sigma_v = sigma_v.to(u.km / u.s)

        sigma_z =  (1 + z) * sigma_v / c
        self.sigma_z = sigma_z.to(u.dimensionless_unscaled)
        return
    
    def add_galaxies(self, galaxies, update_galaxies = True):
        """
            Adds galaxies to the group. This method can be used after identifying candidate groups to populate them with their member galaxies.
        """
        members = set()
        for _, galaxy in enumerate(galaxies):
            member = MemberGalaxy(galaxy, self)
            members.add(member)
            if update_galaxies:
                galaxy.associated_groups.add(self)  # Add this group to each galaxy's set of associated groups
        self._member_galaxies |= members  # Use set union to add galaxies to the group, ensuring no duplicates
        return
    
    def remove_galaxies(self, galaxies, update_galaxies = True):
        """
            Removes galaxies from the group. This method can be used to remove galaxies from groups if they are found to not meet the criteria for group membership.
        """
        self._member_galaxies -= set(galaxies)  # Use set difference to remove galaxies from the group
        if update_galaxies:
            for _, galaxy in enumerate(galaxies):
                galaxy.associated_groups.discard(self)  # Remove this group from each galaxy's set of associated groups

    def add_galaxy(self, galaxy, update_galaxy = True):
        """
            Adds a single galaxy to the group. This method can be used after identifying candidate groups to populate them with their member galaxies.
        """
        self.add_galaxies([galaxy], update_galaxy)
        return
    
    def remove_galaxy(self, galaxy, update_galaxy = True):
        """
            Removes a single galaxy from the group. This method can be used to remove galaxies from groups if they are found to not meet the criteria for group membership.
        """
        self.remove_galaxies([galaxy], update_galaxy)
        return
    
    @staticmethod
    def compute_empirical_prior(groups, delta_z = 10**-3):
        """
        This method can be used to compute the empirical prior probability for groups to be present at their location with their given mass.
        The returned value must be normalized after the method is computed for all groups, so that the probabilities across all groups add up to 1.
        """
        normalisation_factor = 0
        for group in groups:
            z_current = group.pos[2]
            m_current = group.m500
            richness_current = group.richness
            similar_groups = {g for g in groups if np.abs(g.pos[2] - z_current) < delta_z and g.m500 == m_current}

            richness_array = np.array([g.richness for g in similar_groups])
            bins = np.arange(richness_array.min(), richness_array.max() + 2) - 0.5
            hist, edges = np.histogram(richness_array, bins=bins)
            bin_centers = (edges[:-1] + edges[1:]) / 2
            non_normalised_prior = lowess(hist, bin_centers, frac = 1, xvals = [richness_current], return_sorted = False)
            group.probability_prior_empirical = non_normalised_prior[0]
            normalisation_factor += group.probability_prior_empirical

        # Normalize the empirical priors
        for group in groups:
            group.probability_prior_empirical /= normalisation_factor

        return groups
    
    def compute_m500_gapper(self):
        """
        This method can be used to compute the mass of the group using the gapper estimator of velocity dispersion, which acts as a proxy for dynamical mass. See Equation 29 of the draft paper.
        """
        v0 = 474.74 * u.km / u.s
        alpha = 3.18
        M0 = 1e14 * u.Msun
        E_z = self.H / H0
        self.m500_gapper = (self.sigma_v_gapper / v0)**alpha * M0 / E_z
        self.m500_gapper = self.m500_gapper.to(u.Msun)
        return self.m500_gapper