# synth_cluster

Synthetic galaxy cluster generator for member catalogs and source injection *by Tamas N. Varga. / GER-LMU-S3, PI-Seitz*

Based on the research paper *Synthetic Galaxy Clusters and Observations Based on Dark Energy Survey Year 3 Data* [2102.10414](https://arxiv.org/abs/2102.10414)

The instructions below are intended for DESC members.
**If you are using public DESC data sets, please follow the instructions on the [DESC Data Portal: data.lsstdesc.org](https://data.lsstdesc.org/).**

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


## Instructions for installation on nersc to work through the example notebooks

```
$python /global/common/software/lsst/common/miniconda/start-kernel-cli.py desc-stack-weekly-latest  # might need to use desc-stack
$cd $PSCRATCH   # I just chose PSCRATCH.. you could use another area
$mkdir cl-area
$export PYTHONUSERBASE=$PWD/cl-area   
$git checkout git@github.com:vargatn/synth_cluster.git
$cd synth_cluster
```

Uncomment line below if you want to install a branch other than main
git checkout <branch we want to install>  # Not needed if main branch being used.
This will cause pip's --user install to use this new directory
```
$python setup.py install --user
$export PATH=$PYTHONUSERBASE/bin:$PATH           # Not necessary for synth_cluster since it doesn't create a bin directory
$export PYTHONPATH=$PYTHONUSERBASE/synth_cluster:$PYTHONPATH   # Makes sure python can find synth_cluster's library
```
To make this available in NERSC Jupyter, update your $HOME/.bashrc

```
$export PYTHONUSERBASE=$PSCRATCH/cl-area
$PYTHONPATH=$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONPATH
```

1) Start up jupyter.nersc.gov and open up the example jupyter notebooks. (with the environment ‘desk-stack’, if you have followed the first line, python /global/common/software/lsst/common/miniconda/start-kernel-cli.py desc-stack-weekly-latest)
2) Note, you will need to update paths since the notebook is currently pointing to some local directory, for example we can update to use: 
redmapper_path = "/global/cfs/cdirs/lsst/shared/xgal/cosmoDC2/addons/redmapper_v1.1.4/cosmoDC2_v1.1.4_redmapper_v0.7.5_clust.h5"
Note, while I can import skysampler after installation of synth_cluster, skysampler_lsst does not seem to be available in the version of the code in master or on the lsst_dev branch. 


## Instructions for installation on your local computer

```
$git clone https://github.com/LSSTDESC/synth_cluster.git
$cd synth_cluster
$python setup.py install
$export PYTHONPATH=$PATH_to_SYNTH_CLUSTER/synth_cluster:$PYTHONPATH
```

## Utility for multiplying PDFs

represented as a set of points or mcmc samples

See this  [tutorial](notebooks/multiply/Multiply_likelihood_in_chain_PART-1.ipynb)
