import numpy as np
import astropy.units as u
from Galaxy import Galaxy
from astropy.coordinates import SkyCoord

class MemberGalaxy(Galaxy):
    """
    A subclass of Galaxy that represents a member galaxy of a group. 
    This class inherits from the Galaxy class and can have additional properties or methods specific to member galaxies if needed. 
    """
    def __init__(self, galaxy, group):
        # Call the constructor of the parent Galaxy class to initialize the position and properties
        super().__init__(galaxy.pos, galaxy.properties, id = galaxy.id, include_groups = False)  
        self.parent_group = group  # Store a reference to the group that this galaxy belongs to
        self.parent_galaxy = galaxy  # Store a reference to the original Galaxy instance that this MemberGalaxy is based on
        return
    
    def compute_probability_density(self):
        """
        This method can be used to compute the probability density of this galaxy being found at its current position relative to the group center. 
        The parent group must have its properties computed before this method can be called.

        The probability density is split into two terms: the radial term, which depends on the projected distance from the group center, 
        and the line-of-sight term, which depends on the velocity difference along the line of sight.
        """
        center_ra = self.parent_group.pos[0]
        center_dec = self.parent_group.pos[1]
        center_z = self.parent_group.pos[2]
        galaxy_ra = self.pos[0]
        galaxy_dec = self.pos[1]
        galaxy_z = self.pos[2]

        # Compute the projected distance from the group center using astropy's SkyCoord for accurate angular separation calculations
        center_coord = SkyCoord(ra = center_ra, dec = center_dec)
        galaxy_coord = SkyCoord(ra = galaxy_ra, dec = galaxy_dec)
        angular_separation = center_coord.separation(galaxy_coord).to(u.rad, equivalencies = u.dimensionless_angles()).value
        delta_z_projected = (galaxy_z - center_z) * np.cos(angular_separation)  # See section 2.3 of draft paper

        # Compute the line-of-sight probability term
        # This term can be modeled as a Gaussian distribution centered on the group redshift, with a width given by the velocity dispersion of the group
        sigma_z = self.parent_group.sigma_z.value
        prob_density_z = np.exp(-0.5 * (delta_z_projected / sigma_z)**2) / (sigma_z * np.sqrt(2 * np.pi))

        # Compute the radial term of the probability density using the NFW profile of the parent group
        r200 = self.parent_group.r200
        Da = self.parent_group.Da
        projected_radius = angular_separation * Da 
        

        r_max = 0.99 * r200  # Avoid divergence at r = r200
        r_array = np.linspace(0, r_max, 1001)
        Sigma_array = self.parent_group.NFW_Sigma(r_array)

        # Sigma_array diverges at r = 0, but the integrand r * Sigma(r) goes to zero, so we set Sigma_array to zero at small r
        r_tol = 1e-3 * u.Mpc  # Set a small tolerance to avoid numerical issues at r = 0
        mask = r_array < r_tol
        Sigma_array[mask] = 0.0 * u.Msun / u.Mpc**2  # Set Sigma to zero for r < r_tol

        # Same concept also applies for the Sigma at the projected radius
        if projected_radius < r_tol:
            Sigma_projected_radius = 0.0 * u.Msun / u.Mpc**2
        else:
            Sigma_projected_radius = self.parent_group.NFW_Sigma(projected_radius)

        # Numerical Integration to compute probability density
        integrand = r_array * Sigma_array
        prob_density_theta = Da * projected_radius * Sigma_projected_radius / (np.trapezoid(integrand, r_array))

        prob_density_total = prob_density_theta * prob_density_z
        self.probability_density_positional = prob_density_total
        return prob_density_total
    
    # Define equality and hashing methods to allow MemberGalaxy instances to be stored in sets and compared based on their unique IDs  
    def __eq__(self, other):
        if not isinstance(other, MemberGalaxy):
            return NotImplemented
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)