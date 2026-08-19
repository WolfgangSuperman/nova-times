import pandas as pd
from astropy.io import ascii
from astropy.table import Table


def read_csv(filename: str) -> Table: #most flexible
    data = ascii.read(filename, encoding='latin-1').to_pandas()

    # clean up
    data["Magnitude"] = pd.to_numeric(data["Magnitude"], errors="coerce")

    return Table.from_pandas(data).group_by("Band")
