"""
Should containe the Gaussian Processes operations


In addition to the feature spaces we should also take into account the average numbers of objects,
e.g. radial number profile (in absolute terms)


"""

import fitsio as fio
import numpy as np
import pandas as pd
import sklearn.neighbors as neighbors
import sklearn.decomposition as decomp
import copy
import glob
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

BADVAL = -9999

ENDIANS = {
    "little": "<",
    "big": ">",
}

import matplotlib as mpl
try:
    import matplotlib.pyplot as plt
except:
    mpl.use("Agg")
    import matplotlib.pyplot as plt

import multiprocessing as mp

from .utils import partition


def get_angle(num, rng):
    angle = rng.uniform(0, np.pi, size=num)
    return angle


def weighted_mean(values, weights):
    average = np.average(values, axis=0, weights=weights)
    return average


def weighted_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, axis=0, weights=weights)
    variance = np.average((values-average)**2, axis=0, weights=weights)
    return np.sqrt(variance)


class BaseContainer(object):

    def __init__(self):
        self.alldata = None
        self.features = None
        self.weights = None

    def construct_features(self, columns, limits=None, logs=None, **kwargs):
        self.columns = columns
        self.limits = limits
        self.logs = logs
        self.features = pd.DataFrame()

        self.inds = np.ones(len(self.alldata), dtype=bool)
        for i, col in enumerate(columns):
            if isinstance(col[1], str):
                res = self.alldata[col[1]]
            else:
                if len(col[1]) == 3:
                    if isinstance(col[1][0], str):
                        col1 = self.alldata[col[1][0]]
                    elif isinstance(col[1][0], (list, tuple)):
                        col1 = self.alldata[col[1][0][0]][:, col[1][0][1]]
                    else:
                        col1 = col[1][0]

                    if isinstance(col[1][1], str):
                        col2 = self.alldata[col[1][1]]
                    elif isinstance(col[1][1], (list, tuple)):
                        col2 = self.alldata[col[1][1][0]][:, col[1][1][1]]
                    else:
                        col2 = col[1][1]

                    if col[1][2] == "-":
                        res = col1 - col2
                    elif col[1][2] == "+":
                        res = col1 + col2
                    elif col[1][2] == "*":
                        res = col1 * col2
                    elif col[1][2] == "/":
                        res = col1 / col2
                    elif col[1][2] == "SQSUM":
                        res = np.sqrt(col1**2. + col2**2.)
                    else:
                        raise KeyError("only + - * / are supported at the moment")

                elif len(col[1]) == 2:
                    res = self.alldata[col[1][0]][:, col[1][1]]
                else:
                    raise KeyError

            self.features[col[0]] = res.astype("float64")
            #
            if limits is not None:
                self.inds &= (self.features[col[0]] > limits[i][0]) & (self.features[col[0]] < limits[i][1])

        self.features = self.features[self.inds]

        try:
            self.weights = self.alldata["WEIGHT"][self.inds]
        except:
            self.weights = pd.Series(data=np.ones(len(self.features)), name="WEIGHT")

        for i, col in enumerate(columns):
            if logs is not None and logs[i]:
              self.features[col[0]] = np.log10(self.features[col[0]])

    def to_kde(self, **kwargs):
        res = KDEContainer(self.features, weights=self.weights)
        return res


class FeatureSpaceContainer(BaseContainer):
    def __init__(self, info):
        """
        This needs to be done first
        """
        BaseContainer.__init__(self)

        self.rcens = info.rcens
        self.redges = info.redges
        self.rareas = info.rareas

        self.survey = info.survey
        self.target = info.target

        self.numprof = info.numprof
        self.samples = info.samples

        valid_elements = np.nonzero([(len(tmp) > 0) for tmp in self.samples])[0].astype(int)
        if len(valid_elements) != len(self.samples):
            self.alldata = pd.concat(np.array(self.samples)[valid_elements]).reset_index(drop=True)
        else:
            self.alldata = pd.concat(self.samples).reset_index(drop=True)

        self.nobj = self.target.nrow

    def surfdens(self, icol=0, scaler=1):
        if self.logs[icol]:
            arr = 10**self.features.values[:, icol]
        else:
            arr = self.features.values[:, icol]
        vals = np.histogram(arr, bins=self.redges, weights=self.weights)[0] / self.nobj / self.rareas * scaler
        return vals

    def downsample(self, nmax=10000, r_key="LOGR", nbins=40, rng=None, **kwargs):
        """Radially balanced downsampling"""

        if rng is None:
            rng = np.random.RandomState()

        rarr = self.features[r_key]
        # rbins = np.sort(rng.uniform(low=rarr.min(), high=rarr.max(), size=nbins+1))
        rbins = np.linspace(rarr.min(), rarr.max(), nbins+1)

        tmp_features = []
        tmp_weights = []
        for i, tmp in enumerate(rbins[:-1]):
            selinds = (self.features[r_key] > rbins[i]) & (self.features[r_key] < rbins[i + 1])
            vals = self.features.loc[selinds]
            ww = self.weights.loc[selinds]

            if len(vals) < nmax:
                tmp_features.append(vals)
                tmp_weights.append(ww)
            else:
                inds = np.arange(len(vals))
                pp = ww / ww.sum()
                chindex = rng.choice(inds, size=nmax, replace=False, p=pp)

                newvals = vals.iloc[chindex]
                newww = ww.iloc[chindex] * len(ww) / nmax

                tmp_features.append(newvals)
                tmp_weights.append(newww)

        features = pd.concat(tmp_features)
        weights = pd.concat(tmp_weights)

        res = KDEContainer(features, weights=weights)
        # res = DualContainer(features.columns, **kwargs)
        # res.set_data(features, weights=weights)
        return res


class DeepFeatureContainer(BaseContainer):
    def __init__(self, data):
        BaseContainer.__init__(self)
        self.alldata = data
        self.weights = pd.Series(data=np.ones(len(self.alldata)), name="WEIGHT")

    @classmethod
    def from_file(cls, fname, flagsel=True):

        if ".fit" in fname:
            _deep = fio.read(fname)
        else:
            _deep = pd.read_hdf(fname, key="data").to_records()

        if flagsel:
            inds = _deep["flags"] == 0
            deep = _deep[inds]
        else:
            deep = _deep
        return cls(deep)


class KDEContainer(object):
    _default_subset_sizes = (2000, 5000, 10000)
    _kernel = "gaussian"
    # _kernel = "tophat"
    _atol = 1e-6
    _rtol = 1e-6
    _breadth_first = False
    _jacobian_matrix = None
    _jacobian_det = None

    def __init__(self, raw_data, weights=None, transform_params=None, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

        self.data = raw_data
        self.columns = raw_data.columns
        if weights is None:
            self.weights = np.ones(len(raw_data), dtype=float)
        else:
            self.weights = weights.astype(float)
        self.ndim = self.data.shape[1]

    def set_seed(self, seed):
        self.rng.seed(seed)

    @staticmethod
    def _weight_multiplicator(arr, weights):
        multiplier = np.round(weights)
        newarr = []
        for i in np.arange(len(arr)):
            for j in np.arange(multiplier[i]):
                newarr.append(arr[i])
        newarr = np.vstack(newarr)
        return newarr

    def shuffle(self):
        self.sample(n=None, frac=1.)

    def sample(self, n=None, frac=1.):
        inds = np.arange(len(self.data))
        print(self.data.shape)
        print(inds.shape)
        tab = pd.DataFrame()
        tab["IND"] = inds
        if n and n > len(tab):
            n = None
        inds = tab.sample(n=n, frac=frac)["IND"].values

        self.data = self.data.iloc[inds].copy().reset_index(drop=True)
        self.weights = self.weights.iloc[inds].copy().reset_index(drop=True)

    def fit_pca(self):
        """Standardize -> PCA -> Standardize"""
        # _data = self.data
        self.mean1 = weighted_mean(self.data, self.weights)
        _data = self.data - self.mean1
        #
        # Add here a PCA weights pre-burner,
        # draw a subset of 100k rows, then multiplicate them according to weights
        # fit the PCA on theose new rows
        subset = self.select_subset(_data, self.weights, nsample=100000)
        # subset = self._weight_multiplicator(subset.values, self.weights)
        self.pca = decomp.PCA()
        self.pca.fit(subset)

        _data = self.pca.transform(_data)
        self.std2 = weighted_std(_data, self.weights)

        rotation_matrix = self.pca.components_
        scale_matrix = np.diag(1. / self.std2)

        # this is the forward transformation from raw data to processed data
        self._jacobian_matrix = np.dot(scale_matrix, rotation_matrix)
        # self._jacobian_matrix = rotation_matrix

        # this is the inverse transformation from processed data to raw data
        self._jacobian_matrix_inv = np.linalg.inv(self._jacobian_matrix)
        # in the KDE we need the Jacobi determinat of the inverse transformation
        self._jacobian_det = np.linalg.det(self._jacobian_matrix_inv)
        # self._jacobian_det = 1.

        self.pca_params = {
            "mean1": self.mean1.copy(),
            "std2": self.std2.copy(),
            "pca": copy.deepcopy(self.pca),
        }

    def pca_transform(self, data):
        # _data = data
        _data = data - self.mean1
        _data = self.pca.transform(_data)
        _data /= self.std2
        return _data

    def pca_inverse_transform(self, data):
        # _data = data
        _data = data * self.std2
        _data = self.pca.inverse_transform(_data)
        _data = _data + self.mean1
        res = pd.DataFrame(_data, columns=self.columns)
        return res

    def standardize_data(self):
        self.fit_pca()
        self._data = self.pca_transform(self.data)
        # self._data = self.data

    def select_subset(self, data, weights, nsample=10000):
        # if nsample > len(data):
            # nsample = len(data)
        indexes = np.arange(len(data))
        ww = weights / weights.sum()
        inds = self.rng.choice(indexes, size=int(nsample), p=ww, replace=True)
        subset = data.iloc[inds]
        return subset

    def construct_kde(self, bandwidth):
        """"""
        self.bandwidth = bandwidth
        self.kde = neighbors.KernelDensity(bandwidth=self.bandwidth, kernel=self._kernel,
                                           atol=self._atol, rtol=self._rtol, breadth_first=self._breadth_first)
        self.kde.fit(self._data, sample_weight=self.weights)

    def random_draw(self, num, rmin=None, rmax=None, rcol="LOGR"):
        """draws random samples from KDE maximum radius"""
        _res = self.kde.sample(n_samples=int(num), random_state=self.rng)
        self.res = self.pca_inverse_transform(_res)
        if (rmin is not None) or (rmax is not None):
            if rmin is None:
                rmin = self.data[rcol].min()

            if rmax is None:
                rmax = self.data[rcol].max()

            # these are the indexes to replace, not the ones to keep...
            inds = (self.res[rcol] > rmax) | (self.res[rcol] < rmin)
            while inds.sum():
                vals = self.kde.sample(n_samples=int(inds.sum()), random_state=self.rng)
                _res[inds, :] = vals
                self.res = self.pca_inverse_transform(_res)
                inds = (self.res[rcol] > rmax) | (self.res[rcol] < rmin)

        self.res = self.pca_inverse_transform(_res)
        return self.res

    def score_samples(self, arr):
        """Assuming that arr is in the data format"""

        arr = self.pca_transform(arr)
        res = self.kde.score_samples(arr)
        return res, self._jacobian_det

    def drop_kde(self):
        self.pca = None
        self.kde = None

    def drop_col(self, colname):
        self.data = self.data.drop(columns=colname)
        self.columns = self.data.columns
        self.ndim = len(self.columns)


##########################################################################

def construct_wide_container(dataloader, settings, nbins=100, nmax=5000, seed=None, drop=None, **kwargs):
    fsc = FeatureSpaceContainer(dataloader)
    fsc.construct_features(**settings)
    # cont = fsc.to_dual(r_normalize=r_normalize)
    cont_small = fsc.downsample(nbins=nbins, nmax=nmax, kwargs=kwargs)
    cont_small.set_seed(seed)
    cont_small.shuffle()
    if drop is not None:
        cont_small.drop_col(drop)
    # cont_small.standardize_data()
    settings = copy.copy(settings)
    settings.update({"container": cont_small})
    return settings


def construct_deep_container(data, settings, seed=None, frac=1., drop=None):
    fsc = DeepFeatureContainer(data)
    fsc.construct_features(**settings)
    cont = fsc.to_kde()
    if drop is not None:
        cont.drop_col(drop)
    cont.set_seed(seed)
    cont.sample(frac=frac)
    # cont.standardize_data()
    settings = copy.copy(settings)
    settings.update({"container": cont})
    return settings

##########################################################################

def make_classifier_infodicts(wide_cr_clust, wide_r_ref, wide_cr_rands,
                              deep_c, deep_smc, columns,
                              nsamples=1e5, nchunks=1, bandwidth=0.1,
                              rmin=None, rmax=None, rcol="LOGR"):

    deep_smc_emu = deep_smc["container"]
    deep_smc_emu.standardize_data()
    deep_smc_emu.construct_kde(bandwidth)

    wide_r_emu = wide_r_ref["container"]
    wide_r_emu.standardize_data()
    wide_r_emu.construct_kde(bandwidth)

    samples_smc = deep_smc_emu.random_draw(nsamples)
    samples_r = wide_r_emu.random_draw(nsamples, rmin=rmin, rmax=rmax)
    samples = pd.merge(samples_smc, samples_r, left_index=True, right_index=True)
    sample_inds = partition(list(samples.index), nchunks)

    deep_smc_emu.drop_kde()
    wide_r_emu.drop_kde()
    infodicts = []
    for i in np.arange(nchunks):
        info = {
            "columns": columns,
            "bandwidth": bandwidth,
            "wide_cr_clust": wide_cr_clust,
            "wide_cr_rands": wide_cr_rands,
            "deep_c": deep_c,
            "wide_r_ref": wide_r_ref,
            "sample": samples.loc[sample_inds[i]],
            "rmin": rmin,
            "rmax": rmax,
        }
        infodicts.append(info)
    return infodicts, samples


def calc_scores2(info):
    scores = pd.DataFrame()
    try:
        columns = info["columns"]
        bandwidth = info["bandwidth"]
        sample = info["sample"]

        scores = pd.DataFrame()

        dc_emu = info["deep_c"]["container"]
        dc_emu.standardize_data()
        dc_emu.construct_kde(bandwidth=bandwidth)
        _score, _jacobian = dc_emu.score_samples(sample[columns["cols_dc"]])
        scores["dc"] = _score
        scores["dc_jac"] = _jacobian
        # scores["dc_jac"] = 1.

        wr_emu = info["wide_r_ref"]["container"]
        wr_emu.standardize_data()
        wr_emu.construct_kde(bandwidth=bandwidth)
        _score, _jacobian = wr_emu.score_samples(sample[columns["cols_wr"]])
        scores["wr"] = _score
        scores["wr_jac"] = _jacobian
        # scores["wr_jac"] = 1.

        wcr_emu = info["wide_cr_clust"]["container"]
        wcr_emu.standardize_data()
        wcr_emu.construct_kde(bandwidth=bandwidth)
        _score, _jacobian = wcr_emu.score_samples(sample[columns["cols_wcr"]])
        scores["wcr_clust"] = _score
        scores["wcr_clust_jac"] = _jacobian


        wcr_emu = info["wide_cr_rands"]["container"]
        wcr_emu.standardize_data()
        wcr_emu.construct_kde(bandwidth=bandwidth)
        _score, _jacobian = wcr_emu.score_samples(sample[columns["cols_wcr"]])
        scores["wcr_rands"] = _score
        scores["wcr_rands_jac"] = _jacobian
        # scores["wcr_jac"] = 1.

    except KeyboardInterrupt:
        pass

    return scores


def run_scores2(infodicts):
    pool = mp.Pool(processes=len(infodicts))
    try:
        pp = pool.map_async(calc_scores2, infodicts)
        # the results here should be a list of score values
        result = pp.get(86400)  # apparently this counters a bug in the exception passing in python.subprocess...
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
        pool.join()
    else:
        pool.close()
        pool.join()

    return pd.concat(result)


def make_naive_infodicts(wide_cr, wide_r, deep_c, deep_smc, columns,
                         nsamples=1e5, nchunks=1, bandwidth=0.1,
                         rmin=None, rmax=None, rcol="LOGR"):
    deep_smc_emu = deep_smc["container"]
    deep_smc_emu.standardize_data()
    deep_smc_emu.construct_kde(bandwidth)

    wide_r_emu = wide_r["container"]
    wide_r_emu.standardize_data()
    wide_r_emu.construct_kde(bandwidth)

    samples_smc = deep_smc_emu.random_draw(nsamples)
    samples_r = wide_r_emu.random_draw(nsamples, rmin=rmin, rmax=rmax)
    samples = pd.merge(samples_smc, samples_r, left_index=True, right_index=True)
    sample_inds = partition(list(samples.index), nchunks)

    deep_smc_emu.drop_kde()
    wide_r_emu.drop_kde()
    infodicts = []
    for i in np.arange(nchunks):
        info = {
            "columns": columns,
            "bandwidth": bandwidth,
            "wide_cr": wide_cr,
            "deep_c": deep_c,
            "wide_r": wide_r,
            "sample": samples.loc[sample_inds[i]],
            "rmin": rmin,
            "rmax": rmax,
        }
        infodicts.append(info)
    return infodicts, samples


def calc_scores(info):
    try:
        columns = info["columns"]
        bandwidth = info["bandwidth"]
        sample = info["sample"]

        scores = pd.DataFrame()

        dc_emu = info["deep_c"]["container"]
        dc_emu.standardize_data()
        dc_emu.construct_kde(bandwidth=bandwidth)
        _score, _jacobian = dc_emu.score_samples(sample[columns["cols_dc"]])
        scores["dc"] = _score
        scores["dc_jac"] = _jacobian
        # scores["dc_jac"] = 1.

        wr_emu = info["wide_r"]["container"]
        wr_emu.standardize_data()
        wr_emu.construct_kde(bandwidth=bandwidth)
        _score, _jacobian = wr_emu.score_samples(sample[columns["cols_wr"]])
        scores["wr"] = _score
        scores["wr_jac"] = _jacobian
        # scores["wr_jac"] = 1.

        wcr_emu = info["wide_cr"]["container"]
        wcr_emu.standardize_data()
        wcr_emu.construct_kde(bandwidth=bandwidth)
        _score, _jacobian = wcr_emu.score_samples(sample[columns["cols_wcr"]])
        scores["wcr"] = _score
        scores["wcr_jac"] = _jacobian
        # scores["wcr_jac"] = 1.

    except KeyboardInterrupt:
        pass

    return scores


def run_scores(infodicts):
    pool = mp.Pool(processes=len(infodicts))
    try:
        pp = pool.map_async(calc_scores, infodicts)
        # the results here should be a list of score values
        result = pp.get(86400)  # apparently this counters a bug in the exception passing in python.subprocess...
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
        pool.join()
    else:
        pool.close()
        pool.join()

    return pd.concat(result)


##############################################################

def read_concentric(score_path_expr, m_factor=20, seed=6):
    fname_scores = np.sort(glob.glob(score_path_expr))
    fname_samples = []
    for _fname in fname_scores:
        fname_samples.append(_fname.replace("scores", "samples"))

    samples = []
    for _fname in fname_samples:
        _tab = fio.read(_fname)
        _tab = pd.DataFrame.from_records(_tab)
        samples.append(_tab)
    samples = pd.concat(samples)

    scores = []
    for _fname in fname_scores:
        _tab = fio.read(_fname)
        _tab = pd.DataFrame.from_records(_tab)
        scores.append(_tab)
    scores = pd.concat(scores)

    dc_score = np.exp(scores["dc"]) * np.abs(scores["dc_jac"])
    wr_score = np.exp(scores["wr"]) * np.abs(scores["wr_jac"])
    wcr_score = np.exp(scores["wcr"]) * np.abs(scores["wcr_jac"])

    rng = np.random.RandomState(seed)
    uniform = rng.uniform(0, 1, len(samples))
    p_proposal = m_factor * dc_score * wr_score
    p_ref = wcr_score

    inds = uniform < p_ref / p_proposal
    resamples = samples[inds]
    return resamples