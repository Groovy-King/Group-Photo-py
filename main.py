import numpy as np
import pandas as pd
import astropy.units as u
import argparse

from Galaxy import Galaxy
from Group import Group
from constants import omg_m, omg_lambda, omg_k, G, H0, h, c

parser = argparse.ArgumentParser()

parser.add_argument("volume", type=int)
parser.add_argument("slice", type=int)

args = parser.parse_args()
vol = args.volume
sli = args.slice

