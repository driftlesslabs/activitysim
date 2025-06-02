# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import argparse
import collections
import itertools
import logging
import numbers
import os
from collections.abc import Iterable
from operator import itemgetter
from pathlib import Path
from typing import Optional, TypeVar

import cytoolz as tz
import cytoolz.curried
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def si_units(x, kind="B", digits=3, shift=1000):
    """
    Format a number with SI unit prefixes.

    Parameters
    ----------
    x : float or int
        The value to format.
    kind : str, optional
        The unit suffix (default is 'B').
    digits : int, optional
        Number of digits to round to (default is 3).
    shift : int, optional
        The base for SI units (default is 1000).

    Returns
    -------
    str
        The formatted string with SI prefix and unit.
    """
    #       nano micro milli    kilo mega giga tera peta exa  zeta yotta
    tiers = ["n", "µ", "m", "", "K", "M", "G", "T", "P", "E", "Z", "Y"]

    tier = 3
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x > 0:
        while x > shift and tier < len(tiers):
            x /= shift
            tier += 1
        while x < 1 and tier >= 0:
            x *= shift
            tier -= 1
    return f"{sign}{round(x,digits)} {tiers[tier]}{kind}"


def GB(bytes):
    """
    Format bytes as a string with SI units (gigabytes, etc).

    Parameters
    ----------
    bytes : int
        Number of bytes.

    Returns
    -------
    str
        Formatted string with SI prefix and 'B'.
    """
    return si_units(bytes, kind="B", digits=1)


def SEC(seconds):
    """
    Format seconds as a string with SI units.

    Parameters
    ----------
    seconds : int or float
        Number of seconds.

    Returns
    -------
    str
        Formatted string with SI prefix and 's'.
    """
    return si_units(seconds, kind="s", digits=2)


def INT(x):
    """
    Format an integer with underscores as thousands separators.

    Parameters
    ----------
    x : int
        Integer to format.

    Returns
    -------
    str
        Formatted string with underscores.
    """
    # format int as camel case (e.g. 1000000 vecomes '1_000_000')
    negative = x < 0
    x = abs(int(x))
    result = ""
    while x >= 1000:
        x, r = divmod(x, 1000)
        result = "_%03d%s" % (r, result)
    result = "%d%s" % (x, result)

    return f"{'-' if negative else ''}{result}"


def delete_files(file_list, trace_label):
    """
    Delete files in file_list.

    Parameters
    ----------
    file_list : str or list of str
        File(s) to delete.
    trace_label : str
        Label for logging.
    """

    file_list = [file_list] if isinstance(file_list, str) else file_list
    for file_path in file_list:
        try:
            if os.path.isfile(file_path):
                logger.debug(f"{trace_label} deleting {file_path}")
                os.unlink(file_path)
        except Exception as e:
            logger.warning(
                f"{trace_label} exception ({e}) trying to delete {file_path}"
            )


def df_size(df):
    """
    Return a string describing the shape and memory usage of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to describe.

    Returns
    -------
    str
        String with shape and memory usage.
    """
    bytes = 0 if df.empty else df.memory_usage(index=True).sum()
    return "%s %s" % (df.shape, GB(bytes))


def iprod(ints):
    """
    Return the product of the ints in the list or tuple as an unlimited precision python int.

    Specifically intended to compute array/buffer size for skims where np.prod might overflow for default dtypes.

    Parameters
    ----------
    ints : list or tuple of ints
        Sequence of integers to multiply.

    Returns
    -------
    int
        Product of the input integers.
    """
    assert len(ints) > 0
    return int(np.prod(ints, dtype=np.int64))


def left_merge_on_index_and_col(left_df, right_df, join_col, target_col):
    """
    Like pandas left merge, but join on both index and a specified join_col.

    Parameters
    ----------
    left_df : pandas.DataFrame
        Left DataFrame, index name assumed to be same as that of right_df.
    right_df : pandas.DataFrame
        Right DataFrame, index name assumed to be same as that of left_df.
    join_col : str
        Name of column to join on (in addition to index values).
    target_col : str
        Name of column from right_df whose joined values should be returned as series.

    Returns
    -------
    pandas.Series
        Series of target_col values with same index as left_df.
    """
    assert left_df.index.name == right_df.index.name

    # want to know name previous index column will have after reset_index
    idx_col = right_df.index.name

    # SELECT target_col FROM full_sample LEFT JOIN unique_sample on idx_col, join_col
    merged = pd.merge(
        left_df[[join_col]].reset_index(),
        right_df[[join_col, target_col]].reset_index(),
        on=[idx_col, join_col],
        how="left",
    )

    merged.set_index(idx_col, inplace=True)

    return merged[target_col]


def reindex(series1, series2):
    """
    Reindex the first series by the second series.

    Parameters
    ----------
    series1 : pandas.Series
        Series to reindex from.
    series2 : pandas.Series
        Series whose values are used as the new index.

    Returns
    -------
    pandas.Series
        Reindexed series.
    """

    result = series1.reindex(series2)
    try:
        result.index = series2.index
    except AttributeError:
        pass
    return result

    # return pd.Series(series1.loc[series2.values].values, index=series2.index)


def reindex_i(series1, series2, dtype=np.int8):
    """
    Version of reindex that replaces missing na values and converts to int.

    Parameters
    ----------
    series1 : pandas.Series
    series2 : pandas.Series
    dtype : type, optional
        Data type to cast to (default is np.int8).

    Returns
    -------
    pandas.Series
        Reindexed and type-cast series.
    """
    return reindex(series1, series2).fillna(0).astype(dtype)


def other_than(groups, bools):
    """
    Construct a Series that has booleans indicating the presence of something- or someone-else with a certain property within a group.

    Parameters
    ----------
    groups : pandas.Series
        A column with the same index as `bools` that defines the grouping of `bools`.
    bools : pandas.Series
        A boolean Series indicating where the property of interest is present.

    Returns
    -------
    pandas.Series
        A boolean Series with the same index as `groups` and `bools`
        indicating whether there is something- or something-else within
        a group with some property (as indicated by `bools`).

    """
    counts = groups[bools].value_counts()
    merge_col = groups.to_frame(name="right")
    pipeline = tz.compose(
        tz.curry(pd.Series.fillna, value=False),
        itemgetter("left"),
        tz.curry(
            pd.DataFrame.merge,
            right=merge_col,
            how="right",
            left_index=True,
            right_on="right",
        ),
        tz.curry(pd.Series.to_frame, name="left"),
    )
    gt0 = pipeline(counts > 0)
    gt1 = pipeline(counts > 1)

    return gt1.where(bools, other=gt0)


def quick_loc_df(loc_list, target_df, attribute=None):
    """
    Faster replacement for target_df.loc[loc_list] or target_df.loc[loc_list][attribute].

    Parameters
    ----------
    loc_list : list-like
        List of locations to select.
    target_df : pandas.DataFrame
        DataFrame to select from.
    attribute : str, optional
        Name of column to return (or None for all columns).

    Returns
    -------
    pandas.DataFrame or pandas.Series
        Selected rows or column.
    """
    if attribute:
        target_df = target_df[[attribute]]

    df = target_df.reindex(loc_list)

    df.index.name = target_df.index.name

    if attribute:
        # return series
        return df[attribute]
    else:
        # return df
        return df


def quick_loc_series(loc_list, target_series):
    """
    Faster replacement for target_series.loc[loc_list].

    Parameters
    ----------
    loc_list : list-like
        List of locations to select.
    target_series : pandas.Series
        Series to select from.

    Returns
    -------
    pandas.Series
        Selected values.
    """

    left_on = "left"

    if isinstance(loc_list, pd.Index):
        left_df = pd.DataFrame({left_on: loc_list.values})
    elif isinstance(loc_list, pd.Series):
        left_df = loc_list.to_frame(name=left_on)
    elif isinstance(loc_list, np.ndarray) or isinstance(loc_list, list):
        left_df = pd.DataFrame({left_on: loc_list})
    else:
        raise RuntimeError(
            "quick_loc_series loc_list of unexpected type %s" % type(loc_list)
        )

    df = pd.merge(
        left_df,
        target_series.to_frame(name="right"),
        left_on=left_on,
        right_index=True,
        how="left",
    )

    # regression test
    # assert list(df.right) == list(target_series.loc[loc_list])

    return df.right


def assign_in_place(df, df2, downcast_int=False, downcast_float=False):
    """
    Update existing row values in df from df2, adding columns to df if they are not there.

    Parameters
    ----------
    df : pandas.DataFrame
        Assignment left-hand-side (dest).
    df2 : pandas.DataFrame
        Assignment right-hand-side (source).
    downcast_int : bool, optional
        If True, downcast int columns if possible.
    downcast_float : bool, optional
        If True, downcast float columns if possible.
    """
    # expect no rows in df2 that are not in df
    assert len(df2.index.difference(df.index)) == 0

    # update common columns in place
    common_columns = df2.columns.intersection(df.columns)
    if len(common_columns) > 0:
        old_dtypes = [df[c].dtype for c in common_columns]
        df.update(df2)

        # avoid needlessly changing int columns to float
        # this is a hack fix for a bug in pandas.update
        # github.com/pydata/pandas/issues/4094
        for c, old_dtype in zip(common_columns, old_dtypes):
            # if both df and df2 column were same type, but result is not
            if (old_dtype == df2[c].dtype) and (df[c].dtype != old_dtype):
                try:
                    df[c] = df[c].astype(old_dtype)
                except ValueError:
                    logger.warning(
                        "assign_in_place changed dtype %s of column %s to %s"
                        % (old_dtype, c, df[c].dtype)
                    )

            if isinstance(old_dtype, pd.api.types.CategoricalDtype):
                continue

            # if both df and df2 column were ints, but result is not
            if (
                np.issubdtype(old_dtype, np.integer)
                and np.issubdtype(df2[c].dtype, np.integer)
                and not np.issubdtype(df[c].dtype, np.integer)
            ):
                try:
                    df[c] = df[c].astype(old_dtype)
                except ValueError:
                    logger.warning(
                        "assign_in_place changed dtype %s of column %s to %s"
                        % (old_dtype, c, df[c].dtype)
                    )

    # add new columns (in order they appear in df2)
    new_columns = [c for c in df2.columns if c not in df.columns]

    df[new_columns] = df2[new_columns]

    for c in new_columns:
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].astype("category")

    auto_opt_pd_dtypes(df, downcast_int, downcast_float, inplace=True)


def auto_opt_pd_dtypes(
    df_: pd.DataFrame, downcast_int=False, downcast_float=False, inplace=False
) -> Optional[pd.DataFrame]:
    """
    Automatically downcast Number dtypes for minimal possible, will not touch other (datetime, str, object, etc).

    Parameters
    ----------
    df_ : pandas.DataFrame
        DataFrame to optimize.
    downcast_int : bool, optional
        If True, downcast int columns if possible.
    downcast_float : bool, optional
        If True, downcast float columns if possible.
    inplace : bool, optional
        If False, will return a copy of input dataset.

    Returns
    -------
    pandas.DataFrame or None
        Optimized DataFrame if inplace is False, otherwise None.
    """
    df = df_ if inplace else df_.copy()

    for col in df.columns:
        dtype = df[col].dtype
        # Skip optimizing floats for precision concerns
        if pd.api.types.is_float_dtype(dtype):
            if not downcast_float:
                continue
            else:
                # there is a bug in pandas to_numeric
                # when convert int and floats gt 16777216
                # https://github.com/pandas-dev/pandas/issues/43693
                # https://github.com/pandas-dev/pandas/issues/23676#issuecomment-438488603
                if df[col].max() >= 16777216:
                    continue
                else:
                    df[col] = pd.to_numeric(df[col], downcast="float")
        # Skip if the column is already categorical
        if pd.api.types.is_categorical_dtype(dtype):
            continue
        # Handle integer types
        if pd.api.types.is_integer_dtype(dtype):
            if not downcast_int:
                continue
            # there is a bug in pandas to_numeric
            # when convert int and floats gt 16777216
            # https://github.com/pandas-dev/pandas/issues/43693
            # https://github.com/pandas-dev/pandas/issues/23676#issuecomment-438488603
            if df[col].max() >= 16777216:
                continue
            else:
                df[col] = pd.to_numeric(df[col], downcast="integer")
                continue
            # Initially thought of using unsigned integers, BUT:
            # There are calculations in asim (e.g., UECs) that expect results in negative values,
            # and operations on two unsigned types will not produce negative values,
            # therefore, we did not use unsigned integers.

    if not inplace:
        return df


def reindex_if_series(values, index):
    """
    Reindex a Series if index is not None and values is a Series.

    Parameters
    ----------
    values : pandas.Series or other
        Values to reindex if necessary.
    index : pandas.Index or None
        Index to reindex to.

    Returns
    -------
    pandas.Series or other
        Reindexed values or original values.
    """
    if index is not None:
        return values

    if isinstance(values, pd.Series):
        assert len(set(values.index).intersection(index)) == len(index)

        if all(values.index != index):
            return values.reindex(index=index)


def df_from_dict(values, index=None):
    """
    Create a DataFrame from a dictionary of values, reindexing Series if needed.

    Parameters
    ----------
    values : dict
        Dictionary of column names to values.
    index : pandas.Index, optional
        Index to use for the DataFrame.

    Returns
    -------
    pandas.DataFrame
        Constructed DataFrame.
    """
    # If value object is a series and has out of order index, reindex it
    values = {k: reindex_if_series(v, index) for k, v in values.items()}

    df = pd.DataFrame.from_dict(values)
    if index is not None:
        df.index = index

    # 2x slower but users less peak RAM
    # df = pd.DataFrame(index = index)
    # for c in values.keys():
    #     df[c] = values[c]
    #     del values[c]

    return df


# for disaggregate accessibilities


def ordered_load(
    stream, Loader=yaml.SafeLoader, object_pairs_hook=collections.OrderedDict
):
    """
    Load YAML with ordered mapping.

    Parameters
    ----------
    stream : file-like
        YAML stream to load.
    Loader : yaml.Loader, optional
        YAML loader class.
    object_pairs_hook : callable, optional
        Callable for ordered mapping.

    Returns
    -------
    object
        Loaded YAML object.
    """
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    return yaml.load(stream, OrderedLoader)


def named_product(**d):
    """
    Cartesian product of named arguments, yielding dicts.

    Parameters
    ----------
    **d : dict
        Named arguments with iterable values.

    Yields
    ------
    dict
        Dictionary for each combination.
    """
    names = d.keys()
    vals = d.values()
    for res in itertools.product(*vals):
        yield dict(zip(names, res))


def recursive_replace(obj, search, replace):
    """
    Recursively replace values in a nested structure.

    Parameters
    ----------
    obj : object
        Object to search.
    search : object
        Value to search for.
    replace : object
        Value to replace with.

    Returns
    -------
    object
        Modified object.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = recursive_replace(v, search, replace)
    if isinstance(obj, list):
        obj = [replace if x == search else x for x in obj]
    if search == obj:
        obj = replace
    return obj


T = TypeVar("T")


def suffix_tables_in_settings(
    model_settings: T,
    suffix: str = "proto_",
    tables: Iterable[str] = ("persons", "households", "tours", "persons_merged"),
) -> T:
    """
    Add a suffix to table names in model settings.

    Parameters
    ----------
    model_settings : dict or PydanticBase
        Model settings to modify.
    suffix : str, optional
        Suffix to add (default is 'proto_').
    tables : iterable of str, optional
        Table names to replace.

    Returns
    -------
    Same type as model_settings
        Modified model settings.
    """
    if not isinstance(model_settings, dict):
        model_settings_type = type(model_settings)
        model_settings = model_settings.dict()
    else:
        model_settings_type = None

    for k in tables:
        model_settings = recursive_replace(model_settings, k, suffix + k)

    if model_settings_type is not None:
        model_settings = model_settings_type.model_validate(model_settings)

    return model_settings


def suffix_expressions_df_str(
    df, suffix="proto_", tables=["persons", "households", "tours", "persons_merged"]
):
    """
    Add a suffix to table names in the 'expression' column of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with an 'expression' column.
    suffix : str, optional
        Suffix to add (default is 'proto_').
    tables : list of str, optional
        Table names to replace.

    Returns
    -------
    pandas.DataFrame
        Modified DataFrame.
    """
    for k in tables:
        df["expression"] = df.expression.str.replace(k, suffix + k)
    return df


def parse_suffix_args(args):
    """
    Parse command-line arguments for suffixing tables.

    Parameters
    ----------
    args : str
        Argument string to parse.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="file name")
    parser.add_argument("-s", "--SUFFIX", "-s", help="suffix to replace root targets")
    parser.add_argument(
        "-r", "--ROOTS", nargs="*", help="roots be suffixed", default=[]
    )
    return parser.parse_args(args.split())


def concat_suffix_dict(args):
    """
    Convert a dict or BaseModel to a flat list of command-line arguments.

    Parameters
    ----------
    args : dict, BaseModel, or list
        Arguments to convert.

    Returns
    -------
    list
        List of command-line arguments.
    """
    if isinstance(args, BaseModel):
        args = args.dict()
        if "source_file_paths" in args:
            del args["source_file_paths"]
    if isinstance(args, dict):
        args = sum([["--" + k, v] for k, v in args.items()], [])
    if isinstance(args, list):
        args = list(flatten(args))
    return args


def flatten(lst):
    """
    Flatten a nested list.

    Parameters
    ----------
    lst : list
        List to flatten.

    Yields
    ------
    object
        Flattened items.
    """
    for sublist in lst:
        if isinstance(sublist, list):
            for item in sublist:
                yield item
        else:
            yield sublist


def nearest_node_index(node, nodes):
    """
    Find the index of the node in 'nodes' closest to 'node'.

    Parameters
    ----------
    node : array-like
        Target node.
    nodes : array-like
        Array of nodes to search.

    Returns
    -------
    int
        Index of the nearest node.
    """
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum("ij,ij->i", deltas, deltas)
    return np.argmin(dist_2)


def read_csv(filename):
    """
    Simple read of a CSV file, much faster than pandas.read_csv.

    Parameters
    ----------
    filename : str or Path
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame.
    """
    return csv.read_csv(filename).to_pandas()


def to_csv(df, filename, index=False):
    """
    Simple write of a CSV file, much faster than pandas.DataFrame.to_csv.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to write.
    filename : str or Path
        Path to the output CSV file.
    index : bool, optional
        Whether to write the index (default is False).
    """
    filename = Path(filename)
    if filename.suffix == ".gz":
        with pa.CompressedOutputStream(filename, "gzip") as out:
            csv.write_csv(pa.Table.from_pandas(df, preserve_index=index), out)
    else:
        csv.write_csv(pa.Table.from_pandas(df, preserve_index=index), filename)


def read_parquet(filename):
    """
    Simple read of a parquet file.

    Parameters
    ----------
    filename : str or Path
        Path to the parquet file.

    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame.
    """
    return pq.read_table(filename).to_pandas()


def to_parquet(df, filename, index=False):
    """
    Write a DataFrame to a parquet file.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to write.
    filename : str or Path
        Path to the output parquet file.
    index : bool, optional
        Whether to write the index (default is False).
    """
    filename = Path(filename)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=index), filename)


def latest_file_modification_time(filenames: Iterable[Path]):
    """
    Find the most recent file modification time.

    Parameters
    ----------
    filenames : iterable of Path
        Files to check.

    Returns
    -------
    float
        Most recent modification time (timestamp).
    """
    return max(os.path.getmtime(filename) for filename in filenames)


def oldest_file_modification_time(filenames: Iterable[Path]):
    """
    Find the least recent file modification time.

    Parameters
    ----------
    filenames : iterable of Path
        Files to check.

    Returns
    -------
    float
        Oldest modification time (timestamp).
    """
    return min(os.path.getmtime(filename) for filename in filenames)


def zarr_file_modification_time(zarr_dir: Path):
    """
    Find the most recent file modification time inside a zarr dir.

    Parameters
    ----------
    zarr_dir : Path
        Directory to search.

    Returns
    -------
    float
        Most recent modification time (timestamp).
    """
    t = 0
    for dirpath, dirnames, filenames in os.walk(zarr_dir):
        if os.path.basename(dirpath).startswith(".git"):
            continue
        for n in range(len(dirnames) - 1, -1, -1):
            if dirnames[n].startswith(".git"):
                dirnames.pop(n)
        for f in filenames:
            if f.startswith(".git") or f == ".DS_Store":
                continue
            finame = Path(os.path.join(dirpath, f))
            file_time = os.path.getmtime(finame)
            if file_time > t:
                t = file_time
    if t == 0:
        raise FileNotFoundError(zarr_dir)
    return t


def drop_unused_columns(
    choosers,
    spec,
    locals_d,
    custom_chooser,
    sharrow_enabled=False,
    additional_columns=None,
):
    """
    Drop unused columns from the chooser table, based on the spec and custom_chooser function.

    Parameters
    ----------
    choosers : pandas.DataFrame
        Chooser table.
    spec : pandas.DataFrame
        Model spec DataFrame.
    locals_d : dict
        Local variables dictionary.
    custom_chooser : function
        Custom chooser function.
    sharrow_enabled : bool, optional
        Whether sharrow is enabled (default is False).
    additional_columns : list, optional
        Additional columns to keep.

    Returns
    -------
    pandas.DataFrame
        Chooser table with unused columns dropped.
    """
    # keep only variables needed for spec
    import re

    # define a regular expression to find variables in spec
    pattern = r"[a-zA-Z_][a-zA-Z0-9_]*"

    unique_variables_in_spec = set(
        spec.reset_index()["Expression"].apply(lambda x: re.findall(pattern, x)).sum()
    )

    unique_variables_in_spec |= set(additional_columns or [])

    if locals_d:
        unique_variables_in_spec.add(locals_d.get("orig_col_name", None))
        unique_variables_in_spec.add(locals_d.get("dest_col_name", None))
        if locals_d.get("timeframe") == "trip":
            orig_col_name = locals_d.get("ORIGIN", None)
            dest_col_name = locals_d.get("DESTINATION", None)
            stop_col_name = None
            parking_col_name = locals_d.get("PARKING", None)
            primary_origin_col_name = None
            if orig_col_name is None and "od_skims" in locals_d:
                orig_col_name = locals_d["od_skims"].orig_key
            if dest_col_name is None and "od_skims" in locals_d:
                dest_col_name = locals_d["od_skims"].dest_key
            if stop_col_name is None and "dp_skims" in locals_d:
                stop_col_name = locals_d["dp_skims"].dest_key
            if primary_origin_col_name is None and "dnt_skims" in locals_d:
                primary_origin_col_name = locals_d["dnt_skims"].dest_key
            unique_variables_in_spec.add(orig_col_name)
            unique_variables_in_spec.add(dest_col_name)
            unique_variables_in_spec.add(parking_col_name)
            unique_variables_in_spec.add(primary_origin_col_name)
            unique_variables_in_spec.add(stop_col_name)
            unique_variables_in_spec.add("trip_period")
        # when using trip_scheduling_choice for trup scheduling
        unique_variables_in_spec.add("last_outbound_stop")
        unique_variables_in_spec.add("last_inbound_stop")

    # when sharrow mode, need to keep the following columns in the choosers table
    if sharrow_enabled:
        unique_variables_in_spec.add("out_period")
        unique_variables_in_spec.add("in_period")
        unique_variables_in_spec.add("purpose_index_num")

    if custom_chooser:
        import inspect

        custom_chooser_lines = inspect.getsource(custom_chooser)
        unique_variables_in_spec.update(re.findall(pattern, custom_chooser_lines))

    logger.info("Dropping unused variables in chooser table")

    logger.info(
        "before dropping, the choosers table has {} columns: {}".format(
            len(choosers.columns), choosers.columns
        )
    )

    # keep only variables needed for spec
    choosers = choosers[[c for c in choosers.columns if c in unique_variables_in_spec]]

    logger.info(
        "after dropping, the choosers table has {} columns: {}".format(
            len(choosers.columns), choosers.columns
        )
    )

    return choosers
