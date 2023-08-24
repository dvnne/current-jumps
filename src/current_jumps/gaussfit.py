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

def fit_histogram(histogram, p0):
    hist, bin_edges = histogram[0], histogram[1]
    x = np.array([(bin_edges[i] + bin_edges[i-1]) / 2 for i in range(1, len(bin_edges))])
    return gaussfit(x, hist, p0)

def gaussfit(xdata, ydata, p0):
    # Set Bounds
    ubounds, lbounds = [], []
    for i, p in enumerate(p0):
        ptype = i % 3
        if ptype == 0:
            # Amplitude
            lb = 0
            ub = max(ydata) * 1.5
        elif ptype == 1:
            # Center
            lb = p - 3*p0[i+1]
            ub = p + 3*p0[i+1]
        elif ptype == 2:
            # Width
            lb = 0
            ub = 3 * p
        ubounds.append(ub)
        lbounds.append(lb)
    popt, pcov = optimize.curve_fit(multigauss, xdata, ydata, p0=p0, bounds=(lbounds, ubounds))
    return popt

def plot_fit(popt, xlim):
    ax = plt.gca()
    x = np.linspace(*xlim)
    for i in range(len(popt) // 3):
        amp = popt[3*i]
        mu = popt[3*i + 1]
        sigma = popt[3*i + 2]
        lb="%.3G $\\times$ N(%.3G, %.3G)" % (amp, mu, sigma)
        ax.plot(x, gauss(x, amp, mu, sigma), '--', label=lb)
    ax.plot(x, multigauss(x, *popt), label='fit')
    ax.legend()

if __name__ == '__main__':
    pass
