import numpy as np

def function_OU(x, mu, theta, sigma):
    return theta * (mu - x) + sigma * np.random.randn(1)[0]