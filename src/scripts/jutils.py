""" General utilities to help with sampling, modeling, plotting, etc."""
import jax.numpy as jnp
from jax.scipy.special import erf
from scipy.integrate import cumtrapz

def butterworthFilter(x,loc,smooth):
    # if smooth <0, its a low-pass. if smooth<0, its a high-pass
    return 1./(1. + (x/loc)**smooth)

def powerlaw(xx, alpha, high, low):
    r"""
    Normalized truncated power-law probability
    .. math::
        p(x) = \frac{1 + \alpha}{x_\max^{1 + \alpha} - x_\min^{1 + \alpha}} x^\alpha
    Parameters
    ----------
    xx: float, array-like
        The abscissa values (:math:`x`)
    alpha: float, array-like
        The spectral index of the distribution (:math:`\alpha`)
    high: float, array-like
        The maximum of the distribution (:math:`x_\min`)
    low: float, array-like
        The minimum of the distribution (:math:`x_\max`)
    Returns
    -------
    prob: float, array-like
        The distribution evaluated at `xx`
    """
    norm = jnp.where(alpha==-1,
                     1 / jnp.log(high / low),
                     (1 + alpha) / (high ** (1 + alpha) - low ** (1 + alpha))
                    )
    prob = jnp.power(xx, alpha)
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob

def truncnorm(xx, mu, sigma, high, low):
    r"""
    Truncated normal probability
    .. math::
        p(x) =
        \sqrt{\frac{2}{\pi\sigma^2}}
        \left[\text{erf}\left(\frac{x_\max - \mu}{\sqrt{2}}\right) + \text{erf}\left(\frac{\mu - x_\min}{\sqrt{2}}\right)\right]^{-1}
        \exp\left(-\frac{(\mu - x)^2}{2 \sigma^2}\right)
    Parameters
    ----------
    xx: float, array-like
        The abscissa values (:math:`x`)
    mu: float, array-like
        The mean of the normal distribution (:math:`\mu`)
    sigma: float
        The standard deviation of the distribution (:math:`\sigma`)
    high: float, array-like
        The maximum of the distribution (:math:`x_\min`)
    low: float, array-like
        The minimum of the distribution (:math:`x_\max`)
    Returns
    -------
    prob: float, array-like
        The distribution evaluated at `xx`
    """
    norm = 2 ** 0.5 / jnp.pi ** 0.5 / sigma
    norm /= erf((high - mu) / 2 ** 0.5 / sigma) + erf((mu - low) / 2 ** 0.5 / sigma)
    prob = jnp.exp(-jnp.power(xx - mu, 2) / (2 * sigma ** 2))
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob

def contour_plot(samples,theory,grid_m,grid_z, lognorm=True,
                 contour_kwargs=dict(colors='k')):
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LogNorm, Normalize

    if lognorm:
        n = LogNorm()
    else:
        n = Normalize()

    fig, ax = plt.subplots(figsize=(5,5))
#     k = gaussian_kde(samples)
#     positions = jnp.vstack([grid_m.ravel(), grid_z.ravel()])
#     Z = jnp.reshape(k(positions).T, grid_m.shape)
#     ax.contourf(grid_m,grid_z,Z,norm=n)
    ax.hexbin(samples[0],samples[1],norm=n)
    CS = ax.contour(grid_m, grid_z, theory,norm=n, **contour_kwargs)

    return ax