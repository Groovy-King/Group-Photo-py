import numpy as np
import pandas as pd
import astropy.units as u

from Galaxy import Galaxy
from Group import Group
from loading import load_galaxies

def identify_group_centers(volume, slice):
    galaxies = load_galaxies(volume, slice)
    group_centers = []
    
    return group_centers