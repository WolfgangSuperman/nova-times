import pytest

from nova_times.measure import measure_time
from nova_times.io import read_csv


@pytest.fixture
def test_dataset():
    return read_csv("tests/data/aavso_000-BPZ-067.csv")


class TestMeasureTime:

    def test_returns_dict(self, test_dataset):
        result = measure_time(test_dataset)
        assert isinstance(result, dict)

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
        assert result["t2_mag"] == 11.462
        assert isinstance(result["maximum_mag"], float)

    def test_t2_jd(self, test_dataset):
        result = measure_time(test_dataset)
        assert result["t2_jd"] == 2460573.96765
        assert isinstance(result["maximum_jd"], float)
