from typing import Optional, TypedDict

import numpy as np
from astropy.table import Table
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingRegressor

from nova_times.exceptions import MissingDataError


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
    dataset: Table, band: Optional[str] = None, algorithm: Optional[str] = None, N: Optional[float] = None
) -> TimingData:
    MINIMUM_NUM_DATA = 10
    if band is None:
        band = "V"
    if algorithm is None:
        algorithm = "nearest_point"
    if N is None:
        N = 2.

    mask = dataset.groups.keys["Band"] == band
    singleband_data = dataset.groups[mask]
    if len(singleband_data) < MINIMUM_NUM_DATA:
        raise MissingDataError(
            f"{len(singleband_data)} points in band {band}, {MINIMUM_NUM_DATA} required"
        )

    magnitudes = np.array(singleband_data["Magnitude"])
    jds = np.array(singleband_data["JD"])

    algorithm_func = ALGORITHM_FUNCTIONS[algorithm]

    return algorithm_func(magnitudes, jds, band, N)


def nearest_point(mags: NDArray, jds: NDArray, band: str, N: float) -> TimingData:
    """
    Finds observed maximum brightness.
    Finds observation closest to 'TN'.
    Returns Magnitude and JD of each.
    """
    # maximum
    maximum_mag = min(mags)
    maximum_indx = np.argmin(mags)
    maximum_jd = jds[maximum_indx]

    # TN
    tN_mag_calc = maximum_mag + N
    tN_indx = np.argmin(np.abs(mags - tN_mag_calc))
    tN_mag = mags[tN_indx]
    tN_jd = jds[tN_indx]

    results = TimingData(
        band=band,
        algorithm="nearest_point",
        maximum_jd=maximum_jd,
        maximum_mag=maximum_mag,
        N = str(N),
        tN_mag=tN_mag,
        tN_jd=tN_jd,
    )
    return results


def gradient_boosting_regressor(mags: NDArray, jds: NDArray, band: str, N: float) -> TimingData:

    maximum_mag = min(mags)
    maximum_indx = np.argmin(mags)
    maximum_jd = jds[maximum_indx]

    jds = jds[~np.isnan(mags)]
    mags = mags[~np.isnan(mags)]

    jds = jds[np.argmin(mags) :]
    mags = mags[np.argmin(mags) :]
    jds = jds.reshape(-1, 1)

    # Instead of using all data for JDs, use arange over observed min/max
    # 1-hour resolution = 1/24.
    jds_all: NDArray = np.arange(np.min(jds), np.max(jds), 1 / 24.0)
    # jds_all = np.array(alldata['JD'][alldata['JD']<max(jds)])
    # jds_all = np.asarray(sorted(jds_all[np.argmin(mags):]))
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

    results = TimingData(
        band=band,
        algorithm="GBM",
        maximum_jd=maximum_jd,
        maximum_mag=maximum_mag,
        N = str(N),
        tN_mag=tN_mag,
        tN_jd=tN_jd,
    )

    return results

def interpolation(mags: NDArray, jds: NDArray, band: str, N: float) -> TimingData:

    maximum_mag = min(mags)
    maximum_indx = np.argmin(mags)
    maximum_jd = jds[maximum_indx]

    jds = jds[~np.isnan(mags)]
    mags = mags[~np.isnan(mags)]

    jds = jds[np.argmin(mags) :]
    mags = mags[np.argmin(mags) :]

    # Instead of using all data for JDs, use arange over observed min/max
    # 1-hour resolution = 1/24.
    jds_all: NDArray = np.arange(np.min(jds), np.max(jds), 1 / 24.0)
    # jds_all = np.array(alldata['JD'][alldata['JD']<max(jds)])
    # jds_all = np.asarray(sorted(jds_all[np.argmin(mags):]))
   # jds_all = jds_all.reshape(-1, 1)

    fit = np.interp(jds_all, jds, mags)

    tN_indx = np.argmin(np.abs(fit - (mags.min() + N)))
    tN_mag = fit[tN_indx]
    tN_jd = jds_all[tN_indx]

    results = TimingData(
        band=band,
        algorithm="interpolation",
        maximum_jd=maximum_jd,
        maximum_mag=maximum_mag,
        N = str(N),
        tN_mag=tN_mag,
        tN_jd=tN_jd,
    )

    return results

ALGORITHM_FUNCTIONS = {
    "nearest_point": nearest_point,
    "GBM": gradient_boosting_regressor,
    "interpolation": interpolation
}
