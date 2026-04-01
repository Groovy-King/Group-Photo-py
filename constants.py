import astropy.units as u
# based on reed 2007

omg_m = 0.28  # Omega_m
omg_lambda = 0.72  # Omega_lambda
omg_k = 0.0  # flatness of the universe
G = 4.3e-9 * u.km**2 * u.Mpc / u.Msun / u.s**2
H0 = 69.7 * u.km / u.s / u.Mpc  # Hubble constant in km/s/Mpc
h = H0.value / 100.0  # dimensionless Hubble parameter
c = 299792.458 * u.km / u.s  # km/s speed of light