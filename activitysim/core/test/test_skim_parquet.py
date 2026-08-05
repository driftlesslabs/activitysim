# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from activitysim.core.skim_dataset import _load_skim_dataset_from_sources
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
    assert skim_file._orig_idx is None
    assert skim_file._dest_idx is None

    matrix = skim_file.read_matrix("VALUE")
    np.testing.assert_array_equal(matrix, values)


def test_col_major_dense(tmp_path, zone_ids, values):
    df = _dense_col_major_df(zone_ids, values)
    file_path = tmp_path / "skims.parquet"
    df.to_parquet(file_path, index=False)

    skim_file = ParquetSkimFile(str(file_path))
    assert skim_file.is_dense
    assert skim_file.layout == COL_MAJOR
    assert skim_file._orig_idx is None
    assert skim_file._dest_idx is None

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


def test_empty_parquet_raises(tmp_path):
    file_path = tmp_path / "empty.parquet"
    pd.DataFrame(columns=["orig", "dest", "VALUE"]).to_parquet(
        file_path, index=False
    )

    with pytest.raises(ValueError, match="contains no rows"):
        ParquetSkimFile(file_path)


def test_sharrow_parquet_sources(tmp_path, zone_ids, values):
    first = _dense_row_major_df(zone_ids, values).rename(
        columns={"orig": "from_zone", "dest": "to_zone", "VALUE": "DIST"}
    )
    first["TIME__AM"] = first["DIST"] * 2
    first_path = tmp_path / "first.parquet"
    first.to_parquet(first_path, index=False)

    # A different pair of index-column names confirms each Parquet source is
    # inspected independently before all sources are aligned to otaz/dtaz.
    second = _dense_row_major_df(zone_ids, values).rename(
        columns={"orig": "O", "dest": "D", "VALUE": "DISTBIKE"}
    )
    second_path = tmp_path / "second.parquet"
    second.to_parquet(second_path, index=False)

    time_agnostic_dataset, _ = _load_skim_dataset_from_sources(
        [second_path],
        time_periods=["AM", "PM"],
        max_float_precision=32,
        ignore=None,
    )
    np.testing.assert_array_equal(time_agnostic_dataset.time_period, ["AM", "PM"])

    dataset, omx_handles = _load_skim_dataset_from_sources(
        [first_path, second_path],
        time_periods=["AM", "PM"],
        max_float_precision=32,
        ignore=None,
    )

    assert omx_handles == []
    np.testing.assert_array_equal(dataset.otaz, zone_ids)
    np.testing.assert_array_equal(dataset.dtaz, zone_ids)
    np.testing.assert_array_equal(dataset.DIST, values)
    np.testing.assert_array_equal(dataset.DISTBIKE, values)
    np.testing.assert_array_equal(
        dataset.TIME.sel(time_period="AM"), values * 2
    )
    np.testing.assert_array_equal(
        dataset.TIME.sel(time_period="PM"), np.zeros_like(values)
    )


def test_sharrow_mixed_omx_parquet_sources():
    data_dir = Path(__file__).parent / "los" / "data"
    dataset, omx_handles = _load_skim_dataset_from_sources(
        [data_dir / "z1_taz_skims.omx", data_dir / "z1_taz_skims.parquet"],
        time_periods=["EA", "AM", "MD", "PM", "EV"],
        max_float_precision=32,
        ignore=None,
    )

    try:
        assert {"DIST", "DISTBIKE", "SOV_TIME"} <= set(dataset.data_vars)
        assert float(dataset.DIST.sel(otaz=5, dtaz=7)) == pytest.approx(0.4)
        assert float(dataset.DISTBIKE.sel(otaz=23, dtaz=20)) == pytest.approx(
            2.55
        )
    finally:
        for handle in omx_handles:
            handle.close()
