# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from activitysim.core.skim_parquet import (
    COL_MAJOR,
    ROW_MAJOR,
    SPARSE,
    ParquetSkimFile,
    is_parquet_file,
)


def _dense_row_major_df(zone_ids, values):
    n = len(zone_ids)
    orig = np.repeat(zone_ids, n)
    dest = np.tile(zone_ids, n)
    return pd.DataFrame({"orig": orig, "dest": dest, "VALUE": values.flatten()})


def _dense_col_major_df(zone_ids, values):
    n = len(zone_ids)
    orig = np.tile(zone_ids, n)
    dest = np.repeat(zone_ids, n)
    # column-major order means dest varies slowest
    return pd.DataFrame(
        {"orig": orig, "dest": dest, "VALUE": values.flatten(order="F")}
    )


@pytest.fixture
def zone_ids():
    return np.array([10, 20, 30, 40])


@pytest.fixture
def values(zone_ids):
    n = len(zone_ids)
    return np.arange(n * n, dtype="float32").reshape((n, n))


def test_is_parquet_file():
    assert is_parquet_file("foo.parquet")
    assert is_parquet_file("foo.PARQUET")
    assert is_parquet_file("foo.pq")
    assert not is_parquet_file("foo.omx")
    assert not is_parquet_file("foo.csv")


def test_row_major_dense(tmp_path, zone_ids, values):
    df = _dense_row_major_df(zone_ids, values)
    file_path = tmp_path / "skims.parquet"
    df.to_parquet(file_path, index=False)

    skim_file = ParquetSkimFile(str(file_path))
    assert skim_file.is_dense
    assert skim_file.layout == ROW_MAJOR
    assert skim_file.shape == (4, 4)
    np.testing.assert_array_equal(skim_file.zone_ids, zone_ids)

    matrix = skim_file.read_matrix("VALUE")
    np.testing.assert_array_equal(matrix, values)


def test_col_major_dense(tmp_path, zone_ids, values):
    df = _dense_col_major_df(zone_ids, values)
    file_path = tmp_path / "skims.parquet"
    df.to_parquet(file_path, index=False)

    skim_file = ParquetSkimFile(str(file_path))
    assert skim_file.is_dense
    assert skim_file.layout == COL_MAJOR

    matrix = skim_file.read_matrix("VALUE")
    np.testing.assert_array_equal(matrix, values)


def test_sparse_unsorted(tmp_path, zone_ids, values):
    # omit one od pair, and shuffle the rows, to force sparse handling
    df = _dense_row_major_df(zone_ids, values)
    df = df.drop(df.index[5])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    file_path = tmp_path / "skims.parquet"
    df.to_parquet(file_path, index=False)

    skim_file = ParquetSkimFile(str(file_path))
    assert not skim_file.is_dense
    assert skim_file.layout == SPARSE

    matrix = skim_file.read_matrix("VALUE")
    expected = values.copy()
    # the dropped entry defaults to 0 in the dense reconstruction
    dropped_orig_idx, dropped_dest_idx = 1, 1
    expected[dropped_orig_idx, dropped_dest_idx] = 0
    np.testing.assert_array_equal(matrix, expected)


def test_dense_unsorted_raises(tmp_path, zone_ids, values):
    df = _dense_row_major_df(zone_ids, values)
    # shuffle rows so every od pair is present, but not in row-major or
    # column-major order -- this must raise, since the code should not
    # silently read badly-sorted "dense" data via the optimized path,
    # nor should it silently accept a wrong shape via the sparse path.
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    file_path = tmp_path / "skims.parquet"
    df.to_parquet(file_path, index=False)

    with pytest.raises(ValueError):
        ParquetSkimFile(str(file_path))


def test_multiple_data_columns(tmp_path, zone_ids, values):
    df = _dense_row_major_df(zone_ids, values)
    df["VALUE2"] = df["VALUE"] * 10
    file_path = tmp_path / "skims.parquet"
    df.to_parquet(file_path, index=False)

    skim_file = ParquetSkimFile(str(file_path))
    assert skim_file.data_cols == ["VALUE", "VALUE2"]

    np.testing.assert_array_equal(skim_file.read_matrix("VALUE"), values)
    np.testing.assert_array_equal(skim_file.read_matrix("VALUE2"), values * 10)
