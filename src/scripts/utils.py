""" General utilities to help with sampling, modeling, plotting, etc."""
import numpy as np
from scipy.special import erf
from scipy.integrate import cumtrapz

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
    norm = np.where(alpha==-1,
                     1 / np.log(high / low),
                     (1 + alpha) / (high ** (1 + alpha) - low ** (1 + alpha))
                    )
    prob = np.power(xx, alpha)
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
    norm = 2 ** 0.5 / np.pi ** 0.5 / sigma
    norm /= erf((high - mu) / 2 ** 0.5 / sigma) + erf((mu - low) / 2 ** 0.5 / sigma)
    prob = np.exp(-np.power(xx - mu, 2) / (2 * sigma ** 2))
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob

def inverse_transform_sample(pdf, x_limits, rng, N=1000, **pdf_kwargs):
    """Generates N samples from the supplied pdf using inverse transform sampling
    pdf (callable): the probability distribution function defined on x
    x_limits (iterable, length 2): the bounds between which to evaluate the pdf
        and draw samples because many pdfs don't just go to zero by themselves.
    N (int): the number of samples to draw, defaults to 1k,
    pdf_kwargs (kwargs): any keword argument to be passed to the pdf function.
    """

    # get the CDF of the PDF
    xs = np.linspace(x_limits[0], x_limits[1], num=1000)
    if type(pdf)==type(np.array(1)):
        f_of_xs = pdf
    else:
        f_of_xs = pdf(xs, **pdf_kwargs)
    cdf = cumtrapz(f_of_xs, xs, initial=0)
    cdf /= cdf[-1] # normalize

    # choose random points along that CDF
    y = rng.uniform(size=N)

    # convert those to samples by evaluating the value of x
    # required to get you that value of the CDF
    samples = np.interp(y, cdf, xs)

    return samples