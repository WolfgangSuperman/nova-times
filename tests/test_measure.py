import pytest

from nova_times.exceptions import MissingDataError
from nova_times.measure import measure_time
from nova_times.io import read_csv


@pytest.fixture
def test_dataset():
    return read_csv("tests/data/aavso_000-BPZ-067.csv")


class TestMeasureTime:

    def test_returns_dict(self, test_dataset):
        result = measure_time(test_dataset)
        assert isinstance(result, dict)

    def test_default_band_is_V(self, test_dataset):
        result = measure_time(test_dataset)
        assert result["band"] == "V"

    def test_can_select_different_band(self, test_dataset):
        result = measure_time(test_dataset, band="B")
        assert result["band"] == "B"
        assert result["maximum_mag"] == 11.296

    def test_default_algo_is_nearest_point(self, test_dataset):
        result = measure_time(test_dataset)
        assert result["algorithm"] == "nearest_point"

    def test_raises_when_no_data(self, test_dataset):
        with pytest.raises(MissingDataError) as err:
            measure_time(test_dataset, band="NotARealBand")
        assert "0 points in band NotARealBand" in str(err.value)


class TestNearestPoint:

    def test_maximum_mag(self, test_dataset):
        result = measure_time(test_dataset)
        assert result["maximum_mag"] == 9.42
        assert isinstance(result["maximum_mag"], float)

    def test_maximum_jd(self, test_dataset):
        result = measure_time(test_dataset)
        assert result["maximum_jd"] == 2460566.459444
        assert isinstance(result["maximum_jd"], float)

    def test_t2_mag(self, test_dataset):
        result = measure_time(test_dataset)
        assert result["tN_mag"] == 11.462
        assert isinstance(result["maximum_mag"], float)

    def test_t2_jd(self, test_dataset):
        result = measure_time(test_dataset)
        assert result["tN_jd"] == 2460573.96765
        assert isinstance(result["maximum_jd"], float)
