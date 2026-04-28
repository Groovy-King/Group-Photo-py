import numpy as np
import astropy.units as u
import os
from Group import Group

"""

def build_groups(self, mass_bands):
    for mass in mass_bands:
        core_group = Group(self, m500 = mass * u.Msun)
        core_group.compute_properties()
        self.core_groups.add(core_group) 
    
    return self.core_groups
"""

def build_groups(galaxy, mass_bands):
    """
    This method is the place where groups are built. This helps avoid creating duplicate groups.
    """
    for mass in mass_bands:
        core_group = Group(galaxy, m500 = mass * u.Msun)
        core_group.compute_properties()
        galaxy.core_groups.add(core_group) 
    
    return galaxy.core_groups
