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
        "t2_mag": float,
        "t2_jd": float,
    },
)


def measure_time(
    dataset: Table, band: Optional[str] = None, algorithm: Optional[str] = None
) -> TimingData:
    MINIMUM_NUM_DATA = 10
    if band is None:
        band = "V"
    if algorithm is None:
        algorithm = "nearest_point"

    mask = dataset.groups.keys["Band"] == band
    singleband_data = dataset.groups[mask]
    if len(singleband_data) < MINIMUM_NUM_DATA:
        raise MissingDataError(
            f"{len(singleband_data)} points in band {band}, {MINIMUM_NUM_DATA} required"
        )

    magnitudes = np.array(singleband_data["Magnitude"])
    jds = np.array(singleband_data["JD"])

    algorithm_func = ALGORITHM_FUNCTIONS[algorithm]

    return algorithm_func(magnitudes, jds, band)


def nearest_point(mags: NDArray, jds: NDArray, band: str) -> TimingData:
    """
    Finds observed maximum brightness.
    Finds observation closest to T2.
    Returns Magnitude and JD of each.
    """
    # maximum
    maximum_mag = min(mags)
    maximum_indx = np.argmin(mags)
    maximum_jd = jds[maximum_indx]

    # T2
    t2_mag_calc = maximum_mag + 2
    t2_indx = np.argmin(np.abs(mags - t2_mag_calc))
    t2_mag = mags[t2_indx]
    t2_jd = jds[t2_indx]

    results = TimingData(
        band=band,
        algorithm="nearest_point",
        maximum_jd=maximum_jd,
        maximum_mag=maximum_mag,
        t2_mag=t2_mag,
        t2_jd=t2_jd,
    )
    return results


def gradient_boosting_regressor(mags: NDArray, jds: NDArray, band: str) -> TimingData:

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

    t2_indx = np.argmin(np.abs(fit - (mags.min() + 2)))
    t2_mag = fit[t2_indx]
    t2_jd = jds_all[t2_indx][0]

    results = TimingData(
        band=band,
        algorithm="gradient_boosting_regressor",
        maximum_jd=maximum_jd,
        maximum_mag=maximum_mag,
        t2_mag=t2_mag,
        t2_jd=t2_jd,
    )

    return results

def interpolation(mags: NDArray, jds: NDArray, band: str) -> TimingData:

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

    t2_indx = np.argmin(np.abs(fit - (mags.min() + 2)))
    t2_mag = fit[t2_indx]
    t2_jd = jds_all[t2_indx][0]

    results = TimingData(
        band=band,
        algorithm="interpolation",
        maximum_jd=maximum_jd,
        maximum_mag=maximum_mag,
        t2_mag=t2_mag,
        t2_jd=t2_jd,
    )

    return results

ALGORITHM_FUNCTIONS = {
    "nearest_point": nearest_point,
    "gradient_boosting_regressor": gradient_boosting_regressor,
    "interpolation": interpolation
}
