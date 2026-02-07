# pi.py


import numpy as np


def approx_pi(num_points = 10000):
    num_points = int(num_points)
    
    # Keep track of points inside circle
    num_points_inside = 0
    
    rng = np.random.default_rng()
    for i in range(num_points):
        # Generate random point
        point = rng.uniform(-1, 1, size=2)
        # Check if point is inside unit circle
        if point[0]**2 + point[1]**2 <= 1:
            num_points_inside += 1

    # Estimate pi using number of points inside and outside circle
    pi = 4 * (num_points_inside / num_points)
    return pi


if __name__ == "__main__":
    # Test pi approximation with an increasing number of points
    for num_points in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
        pi = approx_pi(num_points)
        error = abs(np.pi - pi)
        print(int(num_points), "pts: \tpi = ", pi, "\terror = ", error)