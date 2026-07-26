# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
import os

import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

PARQUET_SUFFIXES = (".parquet", ".pq")

# layout tags
ROW_MAJOR = "row_major"
COL_MAJOR = "col_major"
SPARSE = "sparse"


def is_parquet_file(file_path):
    """
    Return True if file_path appears to be a parquet skim file, based on its extension.

    Parameters
    ----------
    file_path : str or Path

    Returns
    -------
    bool
    """
    return os.fspath(file_path).lower().endswith(PARQUET_SUFFIXES)


class ParquetSkimFile:
    """
    Inspect and read a single skim matrix table stored in parquet format.

    The parquet file is expected to have an origin column and a destination column
    (the first two columns in the file, whatever their names), followed by one
    column per named skim matrix (analogous to the matrices in an omx file). The
    origin/destination values are used to determine the (square) shape of the
    matrices, and whether the data is arranged 'densely' (i.e. every combination
    of origin and destination is present exactly once) in row-major or column-major
    order, or is instead 'sparse' (i.e. not every combination is present, or the
    dense data is not sorted in row-major or column-major order).
    """

    def __init__(self, file_path):
        self.file_path = file_path

        parquet_file = pq.ParquetFile(file_path)
        column_names = list(parquet_file.schema_arrow.names)
        if len(column_names) < 3:
            raise ValueError(
                f"parquet skim file {file_path} must have at least 3 columns "
                f"(origin, destination, and at least one data column), "
                f"found {len(column_names)}: {column_names}"
            )

        self.orig_col = column_names[0]
        self.dest_col = column_names[1]
        self.data_cols = column_names[2:]

        od_table = parquet_file.read(columns=[self.orig_col, self.dest_col])
        origins = od_table.column(self.orig_col).to_numpy(zero_copy_only=False)
        destinations = od_table.column(self.dest_col).to_numpy(zero_copy_only=False)

        zone_ids = np.unique(np.concatenate([origins, destinations]))
        self.zone_ids = zone_ids
        self.n_zones = len(zone_ids)

        self.shape = (self.n_zones, self.n_zones)

        n_rows = len(origins)
        self.is_dense = n_rows == self.n_zones * self.n_zones

        zone_index = {z: i for i, z in enumerate(zone_ids)}
        orig_idx = np.fromiter(
            (zone_index[o] for o in origins), dtype=np.int64, count=n_rows
        )
        dest_idx = np.fromiter(
            (zone_index[d] for d in destinations), dtype=np.int64, count=n_rows
        )
        self._orig_idx = orig_idx
        self._dest_idx = dest_idx

        if self.is_dense:
            self.layout = self._detect_dense_layout(orig_idx, dest_idx)
        else:
            self.layout = SPARSE

    def _detect_dense_layout(self, orig_idx, dest_idx):
        """
        Determine whether dense data is arranged in row-major or column-major
        order.  Raises a ValueError if the data is dense but not sorted in
        either of these orders.
        """
        n = self.n_zones

        row_major_orig = np.repeat(np.arange(n), n)
        row_major_dest = np.tile(np.arange(n), n)
        if np.array_equal(orig_idx, row_major_orig) and np.array_equal(
            dest_idx, row_major_dest
        ):
            return ROW_MAJOR

        # col-major orig/dest patterns are the same as row-major dest/orig
        if np.array_equal(orig_idx, row_major_dest) and np.array_equal(
            dest_idx, row_major_orig
        ):
            return COL_MAJOR

        raise ValueError(
            f"parquet skim file {self.file_path} appears to contain dense data "
            f"(one row for every origin-destination pair) but the rows are not "
            f"sorted in row-major or column-major order. Dense parquet skim data "
            f"must be sorted so it can be read efficiently; alternatively, omit "
            f"rows to store the data in (unsorted or sorted) sparse format."
        )

    def read_matrix(self, column_name, dtype=None):
        """
        Read a single named skim matrix from the parquet file as a dense 2D array.

        Parameters
        ----------
        column_name : str
        dtype : dtype convertible, optional

        Returns
        -------
        np.ndarray, shape (n_zones, n_zones)
        """
        table = pq.read_table(self.file_path, columns=[column_name])
        values = table.column(column_name).to_numpy(zero_copy_only=False)
        if dtype is not None:
            values = values.astype(dtype, copy=False)

        n = self.n_zones
        if self.layout == ROW_MAJOR:
            return np.ascontiguousarray(values.reshape(n, n))
        elif self.layout == COL_MAJOR:
            return np.ascontiguousarray(values.reshape(n, n, order="F"))
        else:
            # sparse layout (may or may not be sorted); scatter into dense matrix
            matrix = np.zeros((n, n), dtype=values.dtype)
            matrix[self._orig_idx, self._dest_idx] = values
            return matrix
