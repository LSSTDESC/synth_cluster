# synth_cluster
=================================================

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


## Tutorials
