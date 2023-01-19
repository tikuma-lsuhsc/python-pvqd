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
    for id, fs, x in pvqd.iter_data("/a/"):
        pass
    for id, fs, x, info in pvqd.iter_data(
        "/i/", auxdata_fields=["Gender", "Age"], include_cape_v="severity"
    ):
        pass


def test_read_data(pvqd):
    id = "BL01"
    pvqd.read_data(id, padding=0.01)  # full data file
    types = pvqd.task_types  # audio segment types
    for t in types:
        pvqd.read_data(id, t)
