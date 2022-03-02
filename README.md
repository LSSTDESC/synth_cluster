# synth_cluster

Synthetic galaxy cluster generator for member catalogs and source injection *by Tamas N. Varga. / GER-LMU-S3, PI-Seitz*

Based on the research paper *Synthetic Galaxy Clusters and Observations Based on Dark Energy Survey Year 3 Data* [2102.10414](https://arxiv.org/abs/2102.10414)

The instructions below are intended for DESC members.
**If you are using public DESC data sets, please follow the instructions on the [DESC Data Portal: lsstdesc-portal.nersc.gov](https://lsstdesc-portal.nersc.gov/).**

## Python Package

**skysampler** is a python package which can draw random realizations of survey data, with the special aim
of creating random realizations of galaxy clusters and their surrounding galaxies.

The package:

* Handles wide field survey data to learn the joint feature distribution of detections and galaxies
* Draws mock realizations of cluster line-of-sights

Generating mock observations takes place in a data driven way, i.e. clusters are constructed as they are seen in
the survey, not according to our theoretical models for them. Hence the products are not critically dependent
on our physical assumptions, only on survey conditions.

**DES Y3 oriented version of this package is available at https://github.com/vargatn/skysampler**

## Tutorials

There are a set of jupyter notebooks to illustrate the use-case of this software for DESC DC2 data. These are located in the notebooks folder

1) Data Preparation [snyth_cluster_tutorial-1_preparation](notebooks/synth_cluster_tutorial-1_preparation.ipynb)
2) Emulate and extapolate [snyth_cluster_tutorial-2](notebooks/synth_cluster_tutorial-2.ipynb)
3) Create mock catalog [snyth_cluster_tutorial-3_preparation](notebooks/synth_cluster_tutorial-3_generation.ipynb)
4) Render with source injection pipeline [snyth_cluster_tutorial-4_render](notebooks/snyth_cluster_tutorial-4_render.ipynb)

## Utility for multiplying PDFs
represented as a set of points or mcmc samples

See this  [tutorial](notebooks/multiply/Multiply_likelihood_in_chain_PART-1.ipynb)
