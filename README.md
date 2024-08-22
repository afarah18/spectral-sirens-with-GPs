<p align="center">
<a href="https://github.com/showyourwork/showyourwork">
<img width = "450" src="https://raw.githubusercontent.com/showyourwork/.github/main/images/showyourwork.png" alt="showyourwork"/>
</a>
<br>
<br>
<a href="https://github.com/afarah18/spectral-sirens-with-GPs/actions/workflows/build.yml">
<img src="https://github.com/afarah18/spectral-sirens-with-GPs/actions/workflows/build.yml/badge.svg?branch=main" alt="Article status"/>
</a>
<a href="https://github.com/afarah18/spectral-sirens-with-GPs/raw/main-pdf/arxiv.tar.gz">
<img src="https://img.shields.io/badge/article-tarball-blue.svg?style=flat" alt="Article tarball"/>
</a>
<a href="https://github.com/afarah18/spectral-sirens-with-GPs/raw/main-pdf/ms.pdf">
<img src="https://img.shields.io/badge/article-pdf-blue.svg?style=flat" alt="Read the article"/>
</a>
</p>

An open source scientific article about doing GW spectral siren cosmology with a non-parametric model for the mass distribution of GW sources. It was created using the [showyourwork](https://github.com/showyourwork/showyourwork) workflow. 

All source code for this project is containted within `src/scripts/`. Some are not used within the showyourwork workflow as they are expensive (bias_study.py and nonparametric_inference_fitOm0.py). The outputs of these have instead been published to [zenodo](https://zenodo.org/doi/10.5281/zenodo.10963302) and showyourwork is set up to automatically download them when building the article.

**Abstract:**

Gravitational waves (GWs) from merging compact objects encode direct information about the luminosity distance to the binary. 
When paired with a redshift measurement, this enables standard-siren cosmology: a Hubble diagram can be constructed to directly probe the Universe's expansion.
This can be done in the absence of electromagnetic measurements as features in the mass distribution of GW sources provide self-calibrating redshift measurements without the need for a definite or probabilistic host galaxy association. 
This ``spectral siren'' technique has thus far only been applied with simple parametric representations of the mass distribution, and theoretical predictions for features in the mass distribution are commonly presumed to be fundamental to the measurement. 
However, the use of an inaccurate representation leads to biases in the cosmological inference, an acute problem given the current theoretical uncertainties in the population.
Here, we demonstrate that spectral sirens can accurately infer cosmological parameters without prior assumptions for the shape of the mass distribution.
We apply a flexible, non-parametric model for the mass distribution of compact binaries to a simulated catalog of 1,000 GW events, consistent with expectations for the next LVK observing run.
We find that, despite our model's flexibility, both the source mass model and cosmological parameters are correctly reconstructed.
We predict a $5.8\\%$ measurement of $H_0$, keeping all other cosmological parameters fixed, and a $6.4\\%$  measurement of $H(z=0.9)$ when fitting for multiple cosmological parameters ($1\sigma$ uncertainties).
This astrophysically-agnostic spectral siren technique will be essential to arrive at precise and unbiased cosmological constraints from GW source populations.
