import numpy as np
from scipy import stats
from scipy.special import erfinv
from scipy.stats import kurtosis, norm
import warnings


def swtest(x, alpha=0.05):
    """
    Shapiro-Wilk parametric hypothesis test of composite normality.
    
    Parameters:
    -----------
    x : array-like
        A vector of deviates from an unknown distribution. The observation
        number must exceed 3 and less than 5000.
    alpha : float, optional
        The significance level for the test (default = 0.05).
        Must be between 0 and 1.
    
    Returns:
    --------
    H : int
        0 => Do not reject the null hypothesis at significance level alpha.
        1 => Reject the null hypothesis at significance level alpha.
    pValue : float
        The p-value, or the probability of observing the given result by chance
        given that the null hypothesis is true. Small values cast doubt on the
        validity of the null hypothesis.
    W : float
        The test statistic (non normalized).
    
    Notes:
    ------
    When the series 'X' is Leptokurtic, SWTEST performs the Shapiro-Francia
    test, else (series 'X' is Platykurtic) SWTEST performs the Shapiro-Wilk test.
    
    References:
    -----------
    - Royston P. "Remark AS R94", Applied Statistics (1995), Vol. 44, No. 4, pp. 547-551.
    - Royston P. "A pocket-calculator algorithm for the Shapiro-Francia test for
      non-normality: An application to medicine", Statistics in Medicine (1993a),
      Vol. 12, pp. 181-184.
    """
    
    # Ensure the sample data is a vector
    x = np.asarray(x).flatten()
    
    # Remove missing observations (NaN's)
    x = x[~np.isnan(x)]
    
    # Check sample size
    if len(x) < 3:
        raise ValueError('Sample vector "X" must have at least 3 valid observations.')
    
    if len(x) > 5000:
        warnings.warn('Shapiro-Wilk test might be inaccurate due to large sample size (> 5000).')
    
    # Validate alpha
    if not isinstance(alpha, (int, float)):
        raise TypeError('Significance level "alpha" must be a scalar.')
    if alpha <= 0 or alpha >= 1:
        raise ValueError('Significance level "alpha" must be between 0 and 1.')
    
    # Sort the data
    x = np.sort(x)
    n = len(x)
    
    # Calculate the a's for weights as a function of the m's
    # Using inverse normal distribution
    mtilde = norm.ppf(((np.arange(1, n + 1) - 3/8) / (n + 1/4)))
    
    # Determine whether to use Shapiro-Francia or Shapiro-Wilk test
    kurt = kurtosis(x, fisher=True)  # Excess kurtosis (fisher=True gives excess kurtosis)
    
    if kurt > 0:  # Leptokurtic (equivalent to kurtosis(x) > 3 in MATLAB)
        # Shapiro-Francia test for leptokurtic samples
        weights = mtilde / np.sqrt(np.dot(mtilde, mtilde))
        
        # Calculate W statistic
        W = (np.dot(weights, x) ** 2) / np.dot(x - np.mean(x), x - np.mean(x))
        
        # Royston (1993a, p. 183)
        nu = np.log(n)
        u1 = np.log(nu) - nu
        u2 = np.log(nu) + 2 / nu
        mu = -1.2725 + (1.0521 * u1)
        sigma = 1.0308 - (0.26758 * u2)
        
        newSFstatistic = np.log(1 - W)
        NormalSFstatistic = (newSFstatistic - mu) / sigma
        
        # Compute p-value
        pValue = 1 - norm.cdf(NormalSFstatistic, 0, 1)
        
    else:
        # Shapiro-Wilk test for platykurtic samples
        c = mtilde / np.sqrt(np.dot(mtilde, mtilde))
        u = 1 / np.sqrt(n)
        
        # Polynomial coefficients from Royston (1992, 1993b)
        PolyCoef_1 = np.array([-2.706056, 4.434685, -2.071190, -0.147981, 0.221157])
        PolyCoef_2 = np.array([-3.582633, 5.682633, -1.752461, -0.293762, 0.042981])
        PolyCoef_3 = np.array([-0.0006714, 0.0250540, -0.39978, 0.54400])
        PolyCoef_4 = np.array([-0.0020322, 0.0627670, -0.77857, 1.38220])
        PolyCoef_5 = np.array([0.00389150, -0.083751, -0.31082, -1.5861])
        PolyCoef_6 = np.array([0.00303020, -0.082676, -0.48030])
        PolyCoef_7 = np.array([0.459, -2.273])
        
        weights = np.zeros(n)
        
        # Calculate weights
        weights[n - 1] = np.polyval(PolyCoef_1, u) + c[n - 1]
        weights[0] = -weights[n - 1]
        
        if n > 5:
            weights[n - 2] = np.polyval(PolyCoef_2, u) + c[n - 2]
            weights[1] = -weights[n - 2]
            count = 2  # 0-indexed, so corresponds to index 2 (3rd element)
            phi = (np.dot(mtilde, mtilde) - 2 * mtilde[n - 1] ** 2 - 2 * mtilde[n - 2] ** 2) / \
                  (1 - 2 * weights[n - 1] ** 2 - 2 * weights[n - 2] ** 2)
        else:
            count = 1  # 0-indexed
            phi = (np.dot(mtilde, mtilde) - 2 * mtilde[n - 1] ** 2) / \
                  (1 - 2 * weights[n - 1] ** 2)
        
        # Special case for n = 3
        if n == 3:
            weights[0] = 1 / np.sqrt(2)
            weights[n - 1] = -weights[0]
            phi = 1
        
        # Fill in middle weights
        if n > count + 1:
            weights[count:n - count] = mtilde[count:n - count] / np.sqrt(phi)
        
        # Calculate W statistic
        W = (np.dot(weights, x) ** 2) / np.dot(x - np.mean(x), x - np.mean(x))
        
        # Calculate normalized W and significance level
        newn = np.log(n)
        
        if 4 <= n <= 11:
            mu = np.polyval(PolyCoef_3, n)
            sigma = np.exp(np.polyval(PolyCoef_4, n))
            gam = np.polyval(PolyCoef_7, n)
            newSWstatistic = -np.log(gam - np.log(1 - W))
            
        elif n > 11:
            mu = np.polyval(PolyCoef_5, newn)
            sigma = np.exp(np.polyval(PolyCoef_6, newn))
            newSWstatistic = np.log(1 - W)
            
        elif n == 3:
            mu = 0
            sigma = 1
            newSWstatistic = 0
        
        # Compute p-value
        NormalSWstatistic = (newSWstatistic - mu) / sigma
        pValue = 1 - norm.cdf(NormalSWstatistic, 0, 1)
        
        # Special case for n = 3
        if n == 3:
            pValue = (6 / np.pi) * (np.arcsin(np.sqrt(W)) - np.arcsin(np.sqrt(3 / 4)))
    
    # Determine H based on significance level
    H = int(alpha < pValue)  # Returns 0 if reject null, 1 if fail to reject
    
    return H, pValue, W


# Example usage:
if __name__ == "__main__":
    # Test with normally distributed data
    np.random.seed(42)
    x = np.random.normal(loc=0, scale=1, size=100)
    
    H, p_value, W = swtest(x, alpha=0.05)
    
    print(f"Test Statistic (W): {W:.6f}")
    print(f"P-value: {p_value:.6f}")
    print(f"H: {H}")
    
    if H == 0:
        print("Reject the null hypothesis - data is not normal")
    else:
        print("Fail to reject the null hypothesis - data appears normal")