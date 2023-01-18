from pvqd import PVQD
import pandas as pd
from os import path
import numpy as np
from glob import glob
import re
import pytest


def load_db():
    return PVQD(
        r"C:\Users\Takeshi Ikuma\OneDrive - LSUHSC\data\PVQD",  # olol
    )


@pytest.fixture(scope="module")
def pvqd():
    return load_db()


def test_query(pvqd):

    df = pvqd.query()
    df = pvqd.query(include_cape_v=True)
    df = pvqd.query(include_cape_v=["severity"], rating_stats=["mean"])
    df = pvqd.query(include_grbas="breathiness", rating_stats="mean")
    print(df)


def test_files(pvqd):
    print(pvqd.get_files("/a/"))
    print(
        pvqd.get_files(
            "blue",
            "age",
            include_cape_v="severity",
            include_grbas="grade",
            Gender="male",
        )
    )


def test_iter_data(pvqd):
    for fs, x in pvqd.iter_data("rainbow"):
        pass
    for fs, x, info in pvqd.iter_data("ah", auxdata_fields=["SEX", "AGE", "NORM"]):
        pass
