from apertools import utils
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import rayleigh
import numpy as np

# print(plt.style.available)
# ['Solarize_Light2', '_classic_test_patch', 'bmh', 'classic', 'dark_background', 'fast',
# 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind',
# 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted',
# 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk',
# 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']
mpl.style.use("seaborn")


def make_random_img(N, sigma=1):
    rng = np.random.default_rng()
    real = rng.normal(scale=sigma, size=(N, N))
    imag = rng.normal(scale=sigma, size=(N, N))
    return real + 1j * imag


def phase_distribution_by_looks(img=None, looks=[1, 2, 3], bins=70):
    """Show how multilooking the phase of an SLC changes the
    PDF from uniform to gaussian (with decreasing sigma)
    """
    fig, ax = plt.subplots()
    if img is None:
        img = make_random_img(2000, sigma=1)

    for lk in sorted(looks, reverse=True):
        p = np.angle(utils.take_looks(img, lk, lk, separate_complex=True))
        label = f"{lk} looks" if lk > 1 else "single look"
        ax.hist(p.ravel(), bins=bins, density=True, label=label, alpha=0.5)
    ax.set_title("Phase [rad]")
    ax.set_ylabel("PDF")
    ax.legend()


def plot_phase_pdf(N=1000, sigma=1):
    """Plot the complex circular gaussian real, image
    Show how it leads to uniform phase
    TODO: Rayleigh amplitude??
    """
    img = make_random_img(N, sigma=sigma)
    p = img.ravel()

    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    ax.hist(p.real, bins=50, density=True, label=f"Real")
    ax.hist(p.imag, bins=50, density=True, label=f"Imag")
    ax.legend()

    ax = axes[1]
    ax.hist(np.angle(p), bins=50, density=True, label=f"Phase")
    ax.hist(np.abs(p), bins=50, density=True, label=f"Amplitude")
    x = np.linspace(0, 4.5, 100)
    ax.plot(x, rayleigh.pdf(x), lw=5, alpha=0.6, label="Rayleigh PDF")
    ax.legend()
    return fig, axes
