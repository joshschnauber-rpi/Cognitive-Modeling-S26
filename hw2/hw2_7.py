import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
import seaborn as sb



# Multivariate Normal Distribution
class NormalDistribution:
    def __init__(self, mu, Cov):
        self.mu = np.asarray(mu)
        self.Cov = np.asarray(Cov)


    def multivariate_normal_density(self, x):
        """
        Compute the multivariate normal density.

        Parameters
        ----------
        x : array_like, shape (D,)
            Point at which to evaluate the density.

        Returns
        -------
        float
            The probability density evaluated at x.
        """
        x = np.asarray(x)
        mu = self.mu
        Cov = self.Cov

        D = mu.shape[0]

        # Ensure correct shapes
        if x.shape[0] != D:
            raise ValueError("x and mu must have the same dimension")
        if Cov.shape != (D, D):
            raise ValueError("Cov must be a DxD matrix")

        # Compute determinant and inverse
        det_Sigma = np.linalg.det(Cov)
        if det_Sigma <= 0:
            raise ValueError("Covariance matrix must be positive definite")

        inv_Sigma = np.linalg.inv(Cov)

        # Compute normalization constant
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** D * det_Sigma)

        # Compute exponent
        diff = x - mu
        exponent = -0.5 * diff.T @ inv_Sigma @ diff

        return float(norm_const * np.exp(exponent))


    def rvs(self, shape):
        samples = 1000
        D = np.linalg.cholesky(self.Cov)
        m = D.shape[0]
        n = D.shape[1]

        shape = np.asarray(shape)
        x = np.zeros(np.append(shape, m))

        for i in np.ndindex(shape):
            y = np.random.randn(n)
            x_i = self.mu + D @ y
            x[i] = x_i

        return x


    def log_pdf(self, x):
        return np.log(self.multivariate_normal_density(x))



def test_distribution(mu, Cov, samples):
    # Create distributions
    my_dist = NormalDistribution(mu, Cov)
    control_dist = sps.multivariate_normal(mu, Cov)

    # Get samples
    x = my_dist.rvs(samples)

    # Compare densities at each sample
    err = 0
    for i in range(samples):
        my_d = my_dist.multivariate_normal_density(x[i])
        control_d = control_dist.pdf(x[i])
        diff = my_d - control_d
        err += abs(diff)

    return err


# Test implementation
if __name__ == "__main__":
    rng = np.random.default_rng()


    # Test rvs to see it if looks gaussian
    mu = [10, 2]
    Cov = [[1, 0], [0, 2]]
    my_dist = NormalDistribution(mu, Cov)
    samples = 2000
    x = my_dist.rvs(samples)
    sb.kdeplot(pd.DataFrame({"x":x[:,0], "y":x[:,1]}), x="x", y="y", fill=True)
    plt.show()


    # Spherical Gaussian
    total_err = 0
    total_samples = 0
    for _ in range(20):
        N = np.random.randint(10, 100)
        mu = rng.uniform(-10, 10, size=N)
        Cov = np.diag( np.full(N, rng.uniform(0.1, 10)) )

        samples = 100
        err = test_distribution(mu, Cov, samples)
        
        total_err += err
        total_samples += samples
    
    print("Average Error For Spherical Gaussian Across", total_samples, "Samples:\t", round(total_err / total_samples, 4))
    

    # Diagonal Gaussian
    total_err = 0
    total_samples = 0
    for _ in range(20):
        N = np.random.randint(10, 100)
        mu = rng.uniform(-10, 10, size=N)
        Cov = np.diag( rng.uniform(0.1, 10, size=N) )

        samples = 100
        err = test_distribution(mu, Cov, samples)
        
        total_err += err
        total_samples += samples
    
    print("Average Error For Diagonal Gaussian Across", total_samples, "Samples:\t", round(total_err / total_samples, 4))
    

    # Full-Covariance Gaussian
    total_err = 0
    total_samples = 0
    for _ in range(20):
        N = np.random.randint(10, 100)
        mu = rng.uniform(-10, 10, size=N)
        A = rng.uniform(-3, 3, (N, N))
        Cov = A @ A.T 

        samples = 100
        err = test_distribution(mu, Cov, samples)
        
        total_err += err
        total_samples += samples
    
    print("Average Error For Full-Covariance Gaussian Across", total_samples, "Samples:\t", round(total_err / total_samples, 4))
    