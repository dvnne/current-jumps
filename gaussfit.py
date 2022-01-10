import numpy as np
from scipy import optimize
from scipy.stats import norm
import matplotlib.pyplot as plt

def add_noise(d, sigma):
    noise = sigma * np.random.standard_normal(size=len(d))
    return d + noise

def multigauss(x, *args):
    assert len(args) % 3 == 0
    result = np.zeros(len(x))
    for i in range(len(args) // 3):
        amp = args[3*i]
        mu = args[3*i + 1]
        sigma = args[3*i + 2]
        result = result + gauss(x, amp, mu, sigma)
    return result

def gauss(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu)/sigma)**2)

def derivative(signal, order):
    while order > 0:
        dydx = np.zeros(len(signal))
        order -= 1
        for i in range(1, len(signal) - 1):
            dy = signal[i+1] - signal[i-1]
            dx = 2
            dydx[i] = dy / dx
        dydx[0] = dydx[1]
        dydx[-1] = dydx[-2]
        signal = dydx[:]
    return signal

def multi_gaussfit(histogram, p0):
    hist, bin_edges = histogram[0], histogram[1]
    x = np.array([(bin_edges[i] + bin_edges[i-1]) / 2 for i in range(1, len(bin_edges))])
    popt, pcov = optimize.curve_fit(multigauss, x, hist, p0=p0)
    return popt

def plot_fit(histogram, popt):
    ax = plt.gca()
    hist, bin_edges = histogram[0], histogram[1]
    x = np.array([(bin_edges[i] + bin_edges[i-1]) / 2 for i in range(1, len(bin_edges))])
    for i in range(len(popt) // 3):
        amp = popt[3*i]
        mu = popt[3*i + 1]
        sigma = popt[3*i + 2]
        lb="%.2G $\\times$ N(%.2G, %.2G)" % (amp, mu, sigma)
        ax.plot(x, gauss(x, amp, mu, sigma), '--', label=lb)
    ax.plot(x, multigauss(x, *popt), label='fit')

if __name__ == '__main__':
    pass
