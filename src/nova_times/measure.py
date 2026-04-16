from typing import Optional, TypedDict

import numpy as np
from astropy.table import Table
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingRegressor

from nova_times.exceptions import MissingDataError
import matplotlib.pyplot as plt

from nova_times.viz import viz_dataset

import os

TimingData = TypedDict(
    "TimingData",
    {
        "band": str,
        "algorithm": str,
        "maximum_jd": float,
        "maximum_mag": float,
        "N": float,
        "tN_mag": float,
        "tN_jd": float,
    },
)


def measure_time(
    dataset: Table,
    band: Optional[str] = None,
    algorithm: Optional[str] = None,
    N: Optional[float] = None,
    make_plots: Optional[bool] = None,
    lims: Optional[bool] = None,
    output: Optional[str] = None,
) -> TimingData:
    MINIMUM_NUM_DATA = 10
    if band is None:
        band = "V"
    if algorithm is None:
        algorithm = "nearest_point"
    if N is None:
        N = 2.0
    if make_plots is None:
        make_plots = False
    if lims is None:
        lims = False

    mask = dataset.groups.keys["Band"] == band
    singleband_data = dataset.groups[mask]
    if len(singleband_data) < MINIMUM_NUM_DATA:
        raise MissingDataError(
            f"{len(singleband_data)} points in band {band}, {MINIMUM_NUM_DATA} required"
        )

    magnitudes = np.array(singleband_data["Magnitude"])
    jds = np.array(singleband_data["JD"])

    algorithm_func = ALGORITHM_FUNCTIONS[algorithm]

    return algorithm_func(dataset, magnitudes, jds, band, N, make_plots, lims, output)


def nearest_point(
    dataset: Table,
    mags: NDArray,
    jds: NDArray,
    band: str,
    N: float,
    make_plots: bool,
    lims: bool,
    output: Optional[str],
) -> TimingData:
    """
    Finds observed maximum brightness.
    Finds observation closest to 'TN'.
    Returns Magnitude and JD of each.
    """
    # maximum
    maximum_mag = min(mags)
    mags = mags[~np.isnan(mags)]
    jds = jds[np.argmin(mags) :]
    mags = mags[np.argmin(mags) :]

    # TN
    tN_mag_calc = maximum_mag + N
    tN_indx = np.argmin(np.abs(mags - tN_mag_calc))
    tN_mag = mags[tN_indx]
    tN_jd = jds[tN_indx]

    if make_plots:
        fig, ax = plt.subplots()

        if lims:
            plt_lims = np.array([np.min(jds) - 10, tN_jd + 10])
        else:
            plt_lims = None

        viz_dataset(ax, dataset, band, lims=plt_lims)

        ax.axvline(tN_jd, label="t" + str(N) + " JD", ls="--", color="g")
        ax.axhline(tN_mag, label="t" + str(N) + " mag", ls="--", color="g")
        ax.axvline(np.min(jds), label="max JD", ls="--", color="m")

        # indexing on the maximum_jd variable seems to be going awry at times.
        # sometimes returns a jd that is larger than that   of tN_jd (which is
        # incorrect by definition). fix is just pass the string of np.min(jds)
        # wherever appropriate AFTER we have truncated the time array to be
        # max_time onwards

        plt.legend()
        cwd = os.getcwd()
        if output is None:
            print("please provide a filename to save your lightcurve")
        else:
            plt.savefig(cwd + "/" + output)

    results = TimingData(
        band=band,
        algorithm="nearest_point",
        maximum_jd=np.min(jds),
        maximum_mag=maximum_mag,
        N=N,
        tN_mag=tN_mag,
        tN_jd=tN_jd,
    )
    return results


def gradient_boosting_regressor(
    dataset: Table,
    mags: NDArray,
    jds: NDArray,
    band: str,
    N: float,
    make_plots: bool,
    lims: bool,
    output: Optional[str],
) -> TimingData:

    maximum_mag = min(mags)

    jds = jds[~np.isnan(mags)]
    mags = mags[~np.isnan(mags)]

    jds = jds[np.argmin(mags) :]
    mags = mags[np.argmin(mags) :]
    jds = jds.reshape(-1, 1)

    # Instead of using all data for JDs, use arange over observed min/max
    # 1-hour resolution = 1/24.
    jds_all: NDArray = np.arange(np.min(jds), np.max(jds), 1 / 24.0)
    jds_all = jds_all.reshape(-1, 1)

    if len(jds) < 100:
        gbm = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5
        )
    else:
        gbm = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3
        )

    gbm.fit(jds, mags)

    fit = gbm.predict(jds_all)

    tN_indx = np.argmin(np.abs(fit - (mags.min() + N)))
    tN_mag = fit[tN_indx]
    tN_jd = jds_all[tN_indx][0]

    if make_plots:
        fig, ax = plt.subplots()

        if lims:
            plt_lims = np.array([np.min(jds) - 10, tN_jd + 10])
        else:
            plt_lims = None

        viz_dataset(ax, dataset, band, lims=plt_lims)

        ax.plot(jds_all, fit, ls="-.", color="r", label="fit results", alpha=0.7)
        ax.axvline(tN_jd, label="t" + str(N) + " JD", ls="--", color="g")
        ax.axhline(tN_mag, label="t" + str(N) + " mag", ls="--", color="g")
        ax.axvline(np.min(jds), label="max JD", ls="--", color="m")

        # indexing on the maximum_jd variable seems to be going awry at times.
        # sometimes returns a jd that is larger than that   of tN_jd (which is
        # incorrect by definition). fix is just pass the string of np.min(jds)
        # wherever appropriate AFTER we have truncated the time array to be
        # max_time onwards

        plt.legend()
        cwd = os.getcwd()
        if output is None:
            print("please provide a filename to save your lightcurve")
        else:
            plt.savefig(cwd + "/" + output)

    results = TimingData(
        band=band,
        algorithm="GBM",
        maximum_jd=np.min(jds),
        maximum_mag=maximum_mag,
        N=N,
        tN_mag=tN_mag,
        tN_jd=tN_jd,
    )

    return results


def interpolation(
    dataset: Table,
    mags: NDArray,
    jds: NDArray,
    band: str,
    N: float,
    make_plots: bool,
    lims: bool,
    output: Optional[str],
) -> TimingData:

    maximum_mag = min(mags)

    jds = jds[~np.isnan(mags)]
    mags = mags[~np.isnan(mags)]

    jds = jds[np.argmin(mags) :]
    mags = mags[np.argmin(mags) :]

    # Instead of using all data for JDs, use arange over observed min/max
    # 1-hour resolution = 1/24.
    jds_all: NDArray = np.arange(np.min(jds), np.max(jds), 1 / 24.0)

    fit = np.interp(jds_all, jds, mags)

    tN_indx = np.argmin(np.abs(fit - (mags.min() + N)))
    tN_mag = fit[tN_indx]
    tN_jd = jds_all[tN_indx]

    cwd = os.getcwd()

    if make_plots:
        fig, ax = plt.subplots()

        if lims:
            plt_lims = np.array([np.min(jds) - 10, tN_jd + 10])
        else:
            plt_lims = None

        viz_dataset(ax, dataset, band, lims=plt_lims)

        ax.plot(jds_all, fit, ls="-.", color="r", label="fit results", alpha=0.7)
        ax.axvline(tN_jd, label="t" + str(N) + " JD", ls="--", color="g")
        ax.axhline(tN_mag, label="t" + str(N) + " mag", ls="--", color="g")
        ax.axvline(np.min(jds), label="max JD", ls="--", color="m")

        # indexing on the maximum_jd variable seems to be going awry at times.
        # sometimes returns a jd that is larger than that   of tN_jd (which is
        # incorrect by definition). fix is just pass the string of np.min(jds)
        # wherever appropriate AFTER we have truncated the time array to be
        # max_time onwards

        plt.legend()
        cwd = os.getcwd()
        if output is None:
            print("please provide a filename to save your lightcurve")
        else:
            plt.savefig(cwd + "/" + output)

    results = TimingData(
        band=band,
        algorithm="interpolation",
        maximum_jd=np.min(jds),
        maximum_mag=maximum_mag,
        N=N,
        tN_mag=tN_mag,
        tN_jd=tN_jd,
    )

    return results


ALGORITHM_FUNCTIONS = {
    "nearest_point": nearest_point,
    "GBM": gradient_boosting_regressor,
    "interpolation": interpolation,
}
