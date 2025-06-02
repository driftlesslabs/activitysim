# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
import time
import warnings
from collections import OrderedDict
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from activitysim.core import (
    assign,
    chunk,
    config,
    configuration,
    logit,
    pathbuilder,
    tracing,
    util,
    workflow,
)
from activitysim.core.configuration.base import ComputeSettings, PydanticBase
from activitysim.core.configuration.logit import (
    BaseLogitComponentSettings,
    LogitNestSpec,
    TemplatedLogitComponentSettings,
)
from activitysim.core.estimation import Estimator
from activitysim.core.fast_eval import fast_eval
from activitysim.core.simulate_consts import (
    ALT_LOSER_UTIL,
    SPEC_DESCRIPTION_NAME,
    SPEC_EXPRESSION_NAME,
    SPEC_LABEL_NAME,
)

logger = logging.getLogger(__name__)

CustomChooser_T = Callable[
    [workflow.State, pd.DataFrame, pd.DataFrame, pd.DataFrame, str],
    tuple[pd.Series, pd.Series],
]


def random_rows(state: workflow.State, df, n):
    """
    Randomly sample up to n rows from a DataFrame.

    Parameters
    ----------
    state : workflow.State
        The workflow state object containing the random number generator.
    df : pandas.DataFrame
        The DataFrame to sample from.
    n : int
        The number of rows to sample.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with up to n randomly sampled rows.
    """
    # only sample if df has more than n rows
    if len(df.index) > n:
        prng = state.get_rn_generator().get_global_rng()
        return df.take(prng.choice(len(df), size=n, replace=False))

    else:
        return df


def uniquify_spec_index(spec: pd.DataFrame):
    """
    Ensure uniqueness of the spec DataFrame index by appending a comment with a duplicate count.

    Parameters
    ----------
    spec : pandas.DataFrame
        The DataFrame whose index will be made unique in-place.
    """
    # uniquify spec index inplace
    # ensure uniqueness of spec index by appending comment with dupe count
    # this allows us to use pandas dot to compute_utilities
    dict = OrderedDict()
    for expr in spec.index:
        dict[assign.uniquify_key(dict, expr, template="{} # ({})")] = expr

    prev_index_name = spec.index.name
    spec.index = list(dict.keys())
    spec.index.name = prev_index_name

    assert spec.index.is_unique


def read_model_alts(state: workflow.State, file_name, set_index=None):
    """
    Read a CSV file of model alternatives into a DataFrame.

    Parameters
    ----------
    state : workflow.State
        The workflow state object.
    file_name : str
        The name of the alternatives file.
    set_index : str or None
        Column to set as index, if provided.

    Returns
    -------
    pandas.DataFrame
        DataFrame of alternatives.
    """
    file_path = state.filesystem.get_config_file_path(file_name)
    df = pd.read_csv(file_path, comment="#")
    if set_index:
        df.set_index(set_index, inplace=True)
    return df


def read_model_spec(filesystem: configuration.FileSystem, file_name: Path | str):
    """
    Read a CSV model specification into a Pandas DataFrame or Series.

    Parameters
    ----------
    filesystem : configuration.FileSystem
        The configuration filesystem object.
    file_name : Path or str
        The name or path of the spec file.

    Returns
    -------
    pandas.DataFrame
        The description column is dropped from the returned data and the
        expression values are set as the table index.
    """
    if isinstance(file_name, Path):
        file_name = str(file_name)
    assert isinstance(file_name, str)
    if not file_name.lower().endswith(".csv"):
        file_name = f"{file_name}.csv"

    file_path = filesystem.get_config_file_path(file_name)

    try:
        spec = pd.read_csv(file_path, comment="#")
    except Exception as err:
        logger.error(f"read_model_spec error reading {file_path}")
        logger.error(f"read_model_spec error {type(err).__name__}: {str(err)}")
        raise (err)

    spec = spec.dropna(subset=[SPEC_EXPRESSION_NAME])

    # don't need description and set the expression to the index
    if SPEC_DESCRIPTION_NAME in spec.columns:
        spec = spec.drop(SPEC_DESCRIPTION_NAME, axis=1)

    spec = spec.set_index(SPEC_EXPRESSION_NAME).fillna(0)

    # ensure uniqueness of spec index by appending comment with dupe count
    # this allows us to use pandas dot to compute_utilities
    uniquify_spec_index(spec)

    if SPEC_LABEL_NAME in spec:
        spec = spec.set_index(SPEC_LABEL_NAME, append=True)
        assert isinstance(spec.index, pd.MultiIndex)

    return spec


def read_model_coefficients(
    filesystem: configuration.FileSystem,
    model_settings: BaseLogitComponentSettings | dict[str, Any] | None = None,
    file_name: Path | str | None = None,
) -> pd.DataFrame:
    """
    Read the coefficient file specified by COEFFICIENTS model setting.

    Parameters
    ----------
    filesystem : configuration.FileSystem
        The configuration filesystem object.
    model_settings : BaseLogitComponentSettings or dict or None
        Model settings containing the COEFFICIENTS key.
    file_name : Path or str or None
        The name or path of the coefficients file.

    Returns
    -------
    pandas.DataFrame
        DataFrame of coefficients indexed by coefficient_name.
    """
    assert isinstance(filesystem, configuration.FileSystem)

    if model_settings is None:
        assert file_name is not None
    else:
        assert file_name is None
        if isinstance(model_settings, BaseLogitComponentSettings) or (
            isinstance(model_settings, PydanticBase)
            and hasattr(model_settings, "COEFFICIENTS")
        ):
            file_name = model_settings.COEFFICIENTS
        else:
            assert (
                "COEFFICIENTS" in model_settings
            ), "'COEFFICIENTS' tag not in model_settings in %s" % model_settings.get(
                "source_file_paths"
            )
            file_name = model_settings["COEFFICIENTS"]
        logger.debug(f"read_model_coefficients file_name {file_name}")

    file_path = filesystem.get_config_file_path(file_name)
    try:
        coefficients = pd.read_csv(file_path, comment="#", index_col="coefficient_name")
    except ValueError:
        logger.exception("Coefficient File Invalid: %s" % str(file_path))
        raise

    if coefficients.index.duplicated().any():
        logger.warning(
            f"duplicate coefficients in {file_path}\n"
            f"{coefficients[coefficients.index.duplicated(keep=False)]}"
        )
        raise RuntimeError(f"duplicate coefficients in {file_path}")

    if coefficients.value.isnull().any():
        logger.warning(
            f"null coefficients in {file_path}\n{coefficients[coefficients.value.isnull()]}"
        )
        raise RuntimeError(f"null coefficients in {file_path}")

    return coefficients


def spec_for_segment(
    state: workflow.State,
    model_settings: dict | None,
    spec_id: str,
    segment_name: str,
    estimator: Estimator | None,
    *,
    spec_file_name: Path | None = None,
    coefficients_file_name: Path | None = None,
) -> pd.DataFrame:
    """
    Select spec for specified segment from omnibus spec containing columns for each segment.

    Parameters
    ----------
    state : workflow.State
        The workflow state object.
    model_settings : dict or None
        Model settings dictionary.
    spec_id : str
        Key for the spec file in model_settings.
    segment_name : str
        Segment name that is also a column name in the spec.
    estimator : Estimator or None
        Estimator object for estimation mode.
    spec_file_name : Path or None
        Optional path to the spec file.
    coefficients_file_name : Path or None
        Optional path to the coefficients file.

    Returns
    -------
    pandas.DataFrame
        Canonical spec file with expressions in index and a single column with utility coefficients.
    """

    if spec_file_name is None:
        spec_file_name = model_settings[spec_id]
    spec = read_model_spec(state.filesystem, file_name=spec_file_name)

    if len(spec.columns) > 1:
        # if spec is segmented
        spec = spec[[segment_name]]
    else:
        # otherwise we expect a single coefficient column
        # doesn't really matter what it is called, but this may catch errors
        assert spec.columns[0] in ["coefficient", segment_name]

    if (
        coefficients_file_name is None
        and isinstance(model_settings, dict)
        and "COEFFICIENTS" in model_settings
    ):
        coefficients_file_name = model_settings["COEFFICIENTS"]

    if coefficients_file_name is None:
        logger.warning(
            f"no coefficient file specified in model_settings for {spec_file_name}"
        )
        try:
            assert (spec.astype(float) == spec).all(axis=None)
        except (ValueError, AssertionError):
            raise RuntimeError(
                f"No coefficient file specified for {spec_file_name} "
                f"but not all spec column values are numeric"
            ) from None

        return spec

    coefficients = read_model_coefficients(
        state.filesystem, file_name=coefficients_file_name
    )

    spec = eval_coefficients(state, spec, coefficients, estimator)

    return spec


def read_model_coefficient_template(
    filesystem: configuration.FileSystem,
    model_settings: dict | TemplatedLogitComponentSettings,
):
    """
    Read the coefficient template specified by COEFFICIENT_TEMPLATE model setting.

    Parameters
    ----------
    filesystem : configuration.FileSystem
        The configuration filesystem object.
    model_settings : dict or TemplatedLogitComponentSettings
        Model settings containing the COEFFICIENT_TEMPLATE key.

    Returns
    -------
    pandas.DataFrame
        DataFrame of the coefficient template indexed by coefficient_name.
    """

    if isinstance(model_settings, dict):
        assert (
            "COEFFICIENT_TEMPLATE" in model_settings
        ), "'COEFFICIENT_TEMPLATE' not in model_settings in %s" % model_settings.get(
            "source_file_paths"
        )
        coefficients_file_name = model_settings["COEFFICIENT_TEMPLATE"]
    else:
        coefficients_file_name = model_settings.COEFFICIENT_TEMPLATE

    file_path = filesystem.get_config_file_path(coefficients_file_name)
    try:
        template = pd.read_csv(file_path, comment="#", index_col="coefficient_name")
    except ValueError:
        logger.exception("Coefficient Template File Invalid: %s" % str(file_path))
        raise

    # by convention, an empty cell in the template indicates that
    # the coefficient name should be propogated to across all segments
    # this makes for a more legible template than repeating the identical coefficient name in each column

    # replace missing cell values with coefficient_name from index
    template = template.where(
        ~template.isnull(),
        np.broadcast_to(template.index.values[:, None], template.shape),
    )

    if template.index.duplicated().any():
        dupes = template[template.index.duplicated(keep=False)].sort_index()
        logger.warning(
            f"duplicate coefficient names in {coefficients_file_name}:\n{dupes}"
        )
        assert not template.index.duplicated().any()

    return template


def dump_mapped_coefficients(state: workflow.State, model_settings):
    """
    Dump the coefficient template DataFrame with mapped coefficient values to CSV files.

    Parameters
    ----------
    state : workflow.State
        The workflow state object.
    model_settings : dict
        Model settings containing COEFFICIENTS and COEFFICIENT_TEMPLATE keys.
    """

    coefficients_df = state.filesystem.read_model_coefficients(model_settings)
    template_df = read_model_coefficient_template(state.filesystem, model_settings)

    for c in template_df.columns:
        template_df[c] = template_df[c].map(coefficients_df.value)

    coefficients_template_file_name = model_settings["COEFFICIENT_TEMPLATE"]
    file_path = state.get_output_file_path(coefficients_template_file_name)
    template_df.to_csv(file_path, index=True)
    logger.info(f"wrote mapped coefficient template to {file_path}")

    coefficients_file_name = model_settings["COEFFICIENTS"]
    file_path = state.get_output_file_path(coefficients_file_name)
    coefficients_df.to_csv(file_path, index=True)
    logger.info(f"wrote raw coefficients to {file_path}")


def get_segment_coefficients(
    filesystem: configuration.FileSystem,
    model_settings: PydanticBase | dict,
    segment_name: str,
):
    """
    Return a dict mapping generic coefficient names to segment-specific coefficient values.

    Parameters
    ----------
    filesystem : configuration.FileSystem
        The configuration filesystem object.
    model_settings : PydanticBase or dict
        Model settings containing COEFFICIENTS and COEFFICIENT_TEMPLATE keys.
    segment_name : str
        The segment name for which to retrieve coefficients.

    Returns
    -------
    dict
        Dictionary mapping generic coefficient names to segment-specific values.
    """
    if isinstance(model_settings, PydanticBase):
        model_settings = model_settings.dict()

    if (
        "COEFFICIENTS" in model_settings
        and "COEFFICIENT_TEMPLATE" in model_settings
        and model_settings["COEFFICIENTS"] is not None
        and model_settings["COEFFICIENT_TEMPLATE"] is not None
    ):
        legacy = False
    elif (
        "COEFFICIENTS" in model_settings and model_settings["COEFFICIENTS"] is not None
    ):
        legacy = "COEFFICIENTS"
        warnings.warn(
            "Support for COEFFICIENTS without COEFFICIENT_TEMPLATE in model settings file will be removed."
            "Use COEFFICIENT and COEFFICIENT_TEMPLATE to support estimation.",
            FutureWarning,
        )
    elif "LEGACY_COEFFICIENTS" in model_settings:
        legacy = "LEGACY_COEFFICIENTS"
        warnings.warn(
            "Support for 'LEGACY_COEFFICIENTS' setting in model settings file will be removed."
            "Use COEFFICIENT and COEFFICIENT_TEMPLATE to support estimation.",
            FutureWarning,
        )
    else:
        raise RuntimeError("No COEFFICIENTS setting in model_settings")

    if legacy:
        constants = config.get_model_constants(model_settings)
        legacy_coeffs_file_path = filesystem.get_config_file_path(
            model_settings[legacy]
        )
        omnibus_coefficients = pd.read_csv(
            legacy_coeffs_file_path, comment="#", index_col="coefficient_name"
        )
        try:
            omnibus_coefficients_segment_name = omnibus_coefficients[segment_name]
        except KeyError:
            logger.error(f"No key {segment_name} found!")
            possible_keys = "\n- ".join(omnibus_coefficients.keys())
            logger.error(f"possible keys include: \n- {possible_keys}")
            raise
        coefficients_dict = assign.evaluate_constants(
            omnibus_coefficients_segment_name, constants=constants
        )

    else:
        coefficients_df = filesystem.read_model_coefficients(model_settings)
        template_df = read_model_coefficient_template(filesystem, model_settings)
        coefficients_col = (
            template_df[segment_name].map(coefficients_df.value).astype(float)
        )

        if coefficients_col.isnull().any():
            # show them the offending lines from interaction_coefficients_file
            logger.warning(
                f"bad coefficients in COEFFICIENTS {model_settings['COEFFICIENTS']}\n"
                f"{coefficients_col[coefficients_col.isnull()]}"
            )
            assert not coefficients_col.isnull().any()

        coefficients_dict = coefficients_col.to_dict()

    return coefficients_dict


def eval_nest_coefficients(
    nest_spec: LogitNestSpec | dict, coefficients: dict, trace_label: str
) -> LogitNestSpec:
    """
    Replace coefficient names in a nest specification with their values from a coefficients dictionary.

    Parameters
    ----------
    nest_spec : LogitNestSpec or dict
        Nest specification tree.
    coefficients : dict
        Dictionary of coefficient values.
    trace_label : str
        Label for tracing/logging.

    Returns
    -------
    LogitNestSpec
        Nest specification with coefficients replaced by values.
    """
    def replace_coefficients(nest: LogitNestSpec):
        if isinstance(nest, dict):
            assert "coefficient" in nest
            coefficient_name = nest["coefficient"]
            if isinstance(coefficient_name, str):
                assert (
                    coefficient_name in coefficients
                ), f"{coefficient_name} not in nest coefficients"
                nest["coefficient"] = coefficients[coefficient_name]

            assert "alternatives" in nest
            for alternative in nest["alternatives"]:
                if isinstance(alternative, dict | LogitNestSpec):
                    replace_coefficients(alternative)
        elif isinstance(nest, LogitNestSpec):
            if isinstance(nest.coefficient, str):
                assert (
                    nest.coefficient in coefficients
                ), f"{nest.coefficient} not in nest coefficients"
                nest.coefficient = coefficients[nest.coefficient]

            for alternative in nest.alternatives:
                if isinstance(alternative, dict | LogitNestSpec):
                    replace_coefficients(alternative)

    if isinstance(coefficients, pd.DataFrame):
        assert "value" in coefficients.columns
        coefficients = coefficients["value"].to_dict()

    if not isinstance(nest_spec, LogitNestSpec):
        nest_spec = LogitNestSpec.model_validate(nest_spec)

    replace_coefficients(nest_spec)

    logit.validate_nest_spec(nest_spec, trace_label)

    return nest_spec


def eval_coefficients(
    state: workflow.State,
    spec: pd.DataFrame,
    coefficients: dict | pd.DataFrame,
    estimator: Estimator | None,
) -> pd.DataFrame:
    """
    Evaluate and apply coefficients to a spec DataFrame.

    Parameters
    ----------
    state : workflow.State
        The workflow state object.
    spec : pandas.DataFrame
        The spec DataFrame with expressions and coefficients.
    coefficients : dict or pandas.DataFrame
        Coefficient values to apply.
    estimator : Estimator or None
        Estimator object for estimation mode.

    Returns
    -------
    pandas.DataFrame
        The spec DataFrame with coefficients evaluated and applied.
    """
    spec = spec.copy()  # don't clobber input spec

    if isinstance(coefficients, pd.DataFrame):
        assert "value" in coefficients.columns
        coefficients = coefficients["value"].to_dict()

    assert isinstance(
        coefficients, dict
    ), "eval_coefficients doesn't grok type of coefficients: %s" % (type(coefficients))

    for c in spec.columns:
        if c == SPEC_LABEL_NAME:
            continue
        spec[c] = (
            spec[c].apply(lambda x: eval(str(x), {}, coefficients)).astype(np.float32)
        )

    sharrow_enabled = state.settings.sharrow
    if sharrow_enabled:
        # keep all zero rows, reduces the number of unique flows to compile and store.
        return spec

    # drop any rows with all zeros since they won't have any effect (0 marginal utility)
    # (do not drop rows in estimation mode as it may confuse the estimation package (e.g. larch)
    zero_rows = (spec == 0).all(axis=1)
    if zero_rows.any():
        if estimator:
            logger.debug(f"keeping {zero_rows.sum()} all-zero rows in SPEC")
        else:
            logger.debug(f"dropping {zero_rows.sum()} all-zero rows from SPEC")
            spec = spec.loc[~zero_rows]

    return spec


def eval_utilities(
    state,
    spec,
    choosers,
    locals_d=None,
    trace_label=None,
    have_trace_targets=False,
    trace_all_rows=False,
    estimator=None,
    trace_column_names=None,
    log_alt_losers=False,
    zone_layer=None,
    spec_sh=None,
    *,
    chunk_sizer,
    compute_settings: ComputeSettings | None = None,
):
    """
    Evaluate a utility function as defined in a spec file.

    Parameters
    ----------
    state : workflow.State
        The workflow state object.
    spec : pandas.DataFrame
        A table of variable specifications and coefficient values.
        Variable expressions should be in the table index and the table
        should have a column for each alternative.
    choosers : pandas.DataFrame
        DataFrame of choosers.
    locals_d : dict or None
        Dictionary of local variables for expression evaluation.
    trace_label : str or None
        Label for tracing/logging.
    have_trace_targets : bool
        Indicates if `choosers` has targets to trace.
    trace_all_rows : bool
        Trace all chooser rows, bypassing tracing.trace_targets.
    estimator : Estimator or None
        Estimator object for estimation mode.
    trace_column_names : str or list of str or None
        Chooser columns to include when tracing expression_values.
    log_alt_losers : bool
        Write out expressions when all alternatives are unavailable.
    zone_layer : str or None
        Specify which zone layer of the skims is to be used by sharrow.
    spec_sh : pandas.DataFrame or None
        Alternative spec for use with sharrow.
    chunk_sizer : ChunkSizer
        ChunkSizer object for logging.
    compute_settings : ComputeSettings or None
        Settings for sharrow. If not given, the default settings are used.

    Returns
    -------
    pandas.DataFrame
        DataFrame of computed utilities for each chooser and alternative.
    """
    start_time = time.time()

    sharrow_enabled = state.settings.sharrow

    expression_values = None

    from .flow import TimeLogger

    timelogger = TimeLogger("simulate")
    sh_util = None
    sh_flow = None
    utilities = None

    if spec_sh is None:
        spec_sh = spec

    if compute_settings is None:
        compute_settings = ComputeSettings()
    if compute_settings.sharrow_skip:
        sharrow_enabled = False

    if sharrow_enabled:
        from .flow import apply_flow  # import inside func to prevent circular imports

        locals_dict = {}
        locals_dict.update(state.get_global_constants())
        if locals_d is not None:
            locals_dict.update(locals_d)
        sh_util, sh_flow, sh_tree = apply_flow(
            state,
            spec_sh,
            choosers,
            locals_dict,
            trace_label,
            sharrow_enabled == "require",
            zone_layer=zone_layer,
            compute_settings=compute_settings,
        )
        utilities = sh_util
        timelogger.mark("sharrow flow", True, logger, trace_label)
    else:
        timelogger.mark("sharrow flow", False)

    # fixme - restore tracing and _check_for_variability

    if utilities is None or estimator or sharrow_enabled == "test":
        trace_label = tracing.extend_trace_label(trace_label, "eval_utils")

        # avoid altering caller's passed-in locals_d parameter (they may be looping)
        locals_dict = assign.local_utilities(state)

        if locals_d is not None:
            locals_dict.update(locals_d)
        globals_dict = {}

        locals_dict["df"] = choosers

        # - eval spec expressions
        if isinstance(spec.index, pd.MultiIndex):
            # spec MultiIndex with expression and label
            exprs = spec.index.get_level_values(SPEC_EXPRESSION_NAME)
        else:
            exprs = spec.index

        expression_values = np.empty((spec.shape[0], choosers.shape[0]))
        chunk_sizer.log_df(trace_label, "expression_values", expression_values)

        i = 0
        with compute_settings.pandas_option_context():
            for expr, coefficients in zip(exprs, spec.values):
                try:
                    with warnings.catch_warnings(record=True) as w:
                        # Cause all warnings to always be triggered.
                        warnings.simplefilter("always")
                        if expr.startswith("@"):
                            expression_value = eval(expr[1:], globals_dict, locals_dict)
                        else:
                            expression_value = fast_eval(choosers, expr)

                        if len(w) > 0:
                            for wrn in w:
                                logger.warning(
                                    f"{trace_label} - {type(wrn).__name__} ({wrn.message}) evaluating: {str(expr)}"
                                )

                except Exception as err:
                    logger.exception(
                        f"{trace_label} - {type(err).__name__} ({str(err)}) evaluating: {str(expr)}"
                    )
                    raise err

                if log_alt_losers:
                    # utils for each alt for this expression
                    # FIXME if we always did tis, we cold uem these and skip np.dot below
                    utils = np.outer(expression_value, coefficients)
                    losers = np.amax(utils, axis=1) < ALT_LOSER_UTIL

                    if losers.any():
                        logger.warning(
                            f"{trace_label} - {sum(losers)} choosers of {len(losers)} "
                            f"with prohibitive utilities for all alternatives for expression: {expr}"
                        )

                expression_values[i] = expression_value
                i += 1

        chunk_sizer.log_df(trace_label, "expression_values", expression_values)

        if estimator:
            df = pd.DataFrame(
                data=expression_values.transpose(),
                index=choosers.index,
                columns=spec.index.get_level_values(SPEC_LABEL_NAME),
            )
            df.index.name = choosers.index.name
            estimator.write_expression_values(df)

        # - compute_utilities
        utilities = np.dot(
            expression_values.transpose(), spec.astype(np.float64).values
        )

        timelogger.mark("simple flow", True, logger=logger, suffix=trace_label)
    else:
        timelogger.mark("simple flow", False)

    utilities = pd.DataFrame(data=utilities, index=choosers.index, columns=spec.columns)
    chunk_sizer.log_df(trace_label, "utilities", utilities)
    timelogger.mark("assemble utilities")

    # sometimes tvpb will drop rows on the fly and we wind up with an empty
    # table of choosers. this will just bypass tracing in that case.
    if (trace_all_rows or have_trace_targets) and (len(choosers) > 0):
        if trace_all_rows:
            trace_targets = pd.Series(True, index=choosers.index)
        else:
            trace_targets = state.tracing.trace_targets(choosers)
            assert trace_targets.any()  # since they claimed to have targets...

        # get int offsets of the trace_targets (offsets of bool=True values)
        offsets = np.nonzero(list(trace_targets))[0]

        # trace sharrow
        # TODO: This block of code is sometimes extremely slow or hangs for no apparent
        #       reason. It is temporarily disabled until the cause can be identified, so
        #       that most tracing can be still be done with sharrow enabled.
        # if sh_flow is not None:
        #     try:
        #         data_sh = sh_flow.load(
        #             sh_tree.replace_datasets(
        #                 df=choosers.iloc[offsets],
        #             ),
        #             dtype=np.float32,
        #         )
        #         expression_values_sh = pd.DataFrame(data=data_sh.T, index=spec.index)
        #     except ValueError:
        #         expression_values_sh = None
        # else:
        expression_values_sh = None

        # get array of expression_values
        # expression_values.shape = (len(spec), len(choosers))
        # data.shape = (len(spec), len(offsets))
        if expression_values is not None:
            data = expression_values[:, offsets]

            # index is utility expressions (and optional label if MultiIndex)
            expression_values_df = pd.DataFrame(data=data, index=spec.index)

            if trace_column_names is not None:
                if isinstance(trace_column_names, str):
                    trace_column_names = [trace_column_names]
                expression_values_df.columns = pd.MultiIndex.from_frame(
                    choosers.loc[trace_targets, trace_column_names]
                )
        else:
            expression_values_df = None

        if expression_values_sh is not None:
            state.tracing.trace_df(
                expression_values_sh,
                tracing.extend_trace_label(trace_label, "expression_values_sh"),
                slicer=None,
                transpose=False,
            )
        if expression_values_df is not None:
            state.tracing.trace_df(
                expression_values_df,
                tracing.extend_trace_label(trace_label, "expression_values"),
                slicer=None,
                transpose=False,
            )

            if len(spec.columns) > 1:
                for c in spec.columns:
                    name = f"expression_value_{c}"

                    state.tracing.trace_df(
                        expression_values_df.multiply(spec[c].values, axis=0),
                        tracing.extend_trace_label(trace_label, name),
                        slicer=None,
                        transpose=False,
                    )
        timelogger.mark("trace", True, logger, trace_label)

    if sharrow_enabled == "test":
        try:
            np.testing.assert_allclose(
                sh_util,
                utilities.values,
                rtol=1e-2,
                atol=1e-6,
                err_msg="utility not aligned",
                verbose=True,
            )
        except AssertionError as err:
            print(err)
            misses = np.where(
                ~np.isclose(sh_util, utilities.values, rtol=1e-2, atol=1e-6)
            )
            _sh_util_miss1 = sh_util[tuple(m[0] for m in misses)]
            _u_miss1 = utilities.values[tuple(m[0] for m in misses)]
            _sh_util_miss1 - _u_miss1
            if len(misses[0]) > sh_util.size * 0.01:
                print(
                    f"big problem: {len(misses[0])} missed close values "
                    f"out of {sh_util.size} ({100*len(misses[0]) / sh_util.size:.2f}%)"
                )
                print(f"{sh_util.shape=}")
                print(misses)
                # load sharrow flow
                # TODO: This block of code is sometimes extremely slow or hangs for no apparent
                #       reason. It is temporarily disabled until the cause can be identified, so
                #       that model does not hang with sharrow enabled.
                # _sh_flow_load = sh_flow.load(sh_tree)
                # print("possible problematic expressions:")
                # for expr_n, expr in enumerate(exprs):
                #     closeness = np.isclose(
                #         _sh_flow_load[:, expr_n], expression_values[expr_n, :]
                #     )
                #     if not closeness.all():
                #         print(
                #             f"  {closeness.sum()/closeness.size:05.1%} [{expr_n:03d}] {expr}"
                #         )
                raise
        except TypeError as err:
            print(err)
            print("sh_util")
            print(sh_util)
            print("utilities")
            print(utilities)
        timelogger.mark("sharrow test", True, logger, trace_label)

    del expression_values
    chunk_sizer.log_df(trace_label, "expression_values", None)

    # no longer our problem - but our caller should re-log this...
    chunk_sizer.log_df(trace_label, "utilities", None)

    end_time = time.time()
    logger.info(
        f"simulate.eval_utils runtime: {timedelta(seconds=end_time - start_time)} {trace_label}"
    )
    timelogger.summary(logger, "simulate.eval_utils timing")
    return utilities


def eval_variables(state: workflow.State, exprs, df, locals_d=None):
    """
    Evaluate a set of variable expressions from a spec in the context
    of a given data table.

    Parameters
    ----------
    state : workflow.State
        The workflow state object.
    exprs : sequence of str
        Expressions to evaluate.
    df : pandas.DataFrame
        DataFrame providing the context for evaluation.
    locals_d : dict or None
        Dictionary of local variables for expression evaluation.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the index of `df` and columns of eval results of `exprs`.
    """

    # avoid altering caller's passed-in locals_d parameter (they may be looping)
    locals_dict = assign.local_utilities(state)
    if locals_d is not None:
        locals_dict.update(locals_d)
    globals_dict = {}

    locals_dict["df"] = df

    def to_array(x):
        if x is None or np.isscalar(x):
            a = np.asanyarray([x] * len(df.index))
        elif isinstance(x, pd.Series):
            # fixme
            # assert x.index.equals(df.index)
            # save a little RAM
            a = x.values
        else:
            a = x

        # FIXME - for performance, it is essential that spec and expression_values
        # FIXME - not contain booleans when dotted with spec values
        # FIXME - or the arrays will be converted to dtype=object within dot()
        if not np.issubdtype(a.dtype, np.number):
            a = a.astype(np.int8)

        return a

    values = OrderedDict()
    for expr in exprs:
        try:
            if expr.startswith("@"):
                expr_values = to_array(eval(expr[1:], globals_dict, locals_dict))
            else:
                expr_values = to_array(fast_eval(df, expr))
            # read model spec should ensure uniqueness, otherwise we should uniquify
            assert expr not in values
            values[expr] = expr_values

        except Exception as err:
            logger.exception(
                f"Variable evaluation failed {type(err).__name__} ({str(err)}) evaluating: {str(expr)}"
            )
            raise err

    values = util.df_from_dict(values, index=df.index)

    return values


def set_skim_wrapper_targets(df, skims):
    """
    Add the dataframe to the SkimWrapper object so that it can be dereferenced
    using the parameters of the skims object.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to which to add skim data as new columns. `df` is modified in-place.
    skims : SkimWrapper or Skim3dWrapper object, or a list or dict of skims
        The skims object is used to contain multiple matrices of
        origin-destination impedances.
    """

    skims = (
        skims
        if isinstance(skims, list)
        else skims.values()
        if isinstance(skims, dict)
        else [skims]
    )

    # assume any object in skims can be treated as a skim
    for skim in skims:
        try:
            skim.set_df(df)
        except AttributeError:
            pass


def compute_nested_exp_utilities(raw_utilities, nest_spec):
    """
    Compute exponentiated nest utilities based on nesting coefficients.

    For nest nodes this is the exponentiated logsum of alternatives adjusted by nesting coefficient.

    Parameters
    ----------
    raw_utilities : pandas.DataFrame
        DataFrame with the raw alternative utilities of all leaves.
    nest_spec : dict
        Nest tree dict from the model spec yaml file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the index of `raw_utilities` and columns for exponentiated leaf and node utilities.
    """
    nested_utilities = pd.DataFrame(index=raw_utilities.index)

    for nest in logit.each_nest(nest_spec, post_order=True):
        name = nest.name

        if nest.is_leaf:
            # leaf_utility = raw_utility / nest.product_of_coefficients
            nested_utilities[name] = (
                raw_utilities[name].astype(float) / nest.product_of_coefficients
            )

        else:
            # nest node
            # the alternative nested_utilities will already have been computed due to post_order
            # this will RuntimeWarning: divide by zero encountered in log
            # if all nest alternative utilities are zero
            # but the resulting inf will become 0 when exp is applied below
            with np.errstate(divide="ignore"):
                nested_utilities[name] = nest.coefficient * np.log(
                    nested_utilities[nest.alternatives].sum(axis=1)
                )

        # exponentiate the utility
        nested_utilities[name] = np.exp(nested_utilities[name])

    return nested_utilities


def compute_nested_probabilities(
    state: workflow.State, nested_exp_utilities, nest_spec, trace_label
):
    """
    Compute nested probabilities for nest leafs and nodes.

    Parameters
    ----------
    state : workflow.State
        The workflow state object.
    nested_exp_utilities : pandas.DataFrame
        DataFrame with the exponentiated nested utilities of all leaves and nodes.
    nest_spec : dict
        Nest tree dict from the model spec yaml file.
    trace_label : str
        Label for tracing/logging.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the index of `nested_exp_utilities` and columns for leaf and node probabilities.
    """

    nested_probabilities = pd.DataFrame(index=nested_exp_utilities.index)

    for nest in logit.each_nest(nest_spec, type="node", post_order=False):
        probs = logit.utils_to_probs(
            state,
            nested_exp_utilities[nest.alternatives],
            trace_label=trace_label,
            exponentiated=True,
            allow_zero_probs=True,
            overflow_protection=False,
        )

        nested_probabilities = pd.concat([nested_probabilities, probs], axis=1)

    return nested_probabilities


def compute_base_probabilities(nested_probabilities, nests, spec):
    """
    Compute base probabilities for nest leaves.

    Parameters
    ----------
    nested_probabilities : pandas.DataFrame
        DataFrame with the nested probabilities for nest leafs and nodes.
    nests : dict
        Nest tree dict from the model spec yaml file.
    spec : pandas.DataFrame
        Simple simulate spec so we can return columns in appropriate order.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the index of `nested_probabilities` and columns for leaf base probabilities.
    """

    base_probabilities = pd.DataFrame(index=nested_probabilities.index)

    for nest in logit.each_nest(nests, type="leaf", post_order=False):
        # skip root: it has a prob of 1 but we didn't compute a nested probability column for it
        ancestors = nest.ancestors[1:]

        base_probabilities[nest.name] = nested_probabilities[ancestors].prod(axis=1)

    # reorder alternative columns to match spec
    # since these are alternatives chosen by column index, order of columns matters
    assert set(base_probabilities.columns) == set(spec.columns)
    base_probabilities = base_probabilities[spec.columns]

    return base_probabilities


def eval_mnl(
    state: workflow.State,
    choosers,
    spec,
    locals_d,
    custom_chooser: CustomChooser_T,
    estimator,
    log_alt_losers=False,
    want_logsums=False,
    trace_label=None,
    trace_choice_name=None,
    trace_column_names=None,
    *,
    chunk_sizer,
    compute_settings: ComputeSettings | None = None,
):
    """
    Run a simulation for when the model spec does not involve alternative
    specific data, e.g. there are no interactions with alternative
    properties and no need to sample from alternatives.

    Each row in spec computes a partial utility for each alternative,
    by providing a spec expression (often a boolean 0-1 trigger)
    and a column of utility coefficients for each alternative.

    We compute the utility of each alternative by matrix-multiplication of eval results
    with the utility coefficients in the spec alternative columns
    yielding one row per chooser and one column per alternative

    Parameters
    ----------
    state : workflow.State
        The workflow state object.
    choosers : pandas.DataFrame
        DataFrame of choosers.
    spec : pandas.DataFrame
        A table of variable specifications and coefficient values.
        Variable expressions should be in the table index and the table
        should have a column for each alternative.
    locals_d : dict or None
        Dictionary of local variables for expression evaluation.
    custom_chooser : function
        Custom alternative to logit.make_choices.
    estimator : Estimator
        Estimator object for estimation mode.
    log_alt_losers : bool
        Write out expressions when all alternatives are unavailable.
    want_logsums : bool
        Whether to return logsums instead of choices.
    trace_label : str or None
        Label for tracing/logging.
    trace_choice_name : str or None
        Column label for trace file csv dump of choices.
    trace_column_names : str or list of str or None
        Chooser columns to include when tracing expression_values.
    chunk_sizer : ChunkSizer
        ChunkSizer object for logging.
    compute_settings : ComputeSettings or None
        Settings for sharrow. If not given, the default settings are used.

    Returns
    -------
    pandas.Series
        Index will be that of `choosers`, values will match the columns of `spec`.
    """

    # FIXME - not implemented because not currently needed
    assert not want_logsums

    trace_label = tracing.extend_trace_label(trace_label, "eval_mnl")
    have_trace_targets = state.tracing.has_trace_targets(choosers)

    if have_trace_targets:
        state.tracing.trace_df(choosers, "%s.choosers" % trace_label)

    utilities = eval_utilities(
        state,
        spec,
        choosers,
        locals_d,
        log_alt_losers=log_alt_losers,
        trace_label=trace_label,
        have_trace_targets=have_trace_targets,
        estimator=estimator,
        trace_column_names=trace_column_names,
        chunk_sizer=chunk_sizer,
        compute_settings=compute_settings,
    )
    chunk_sizer.log_df(trace_label, "utilities", utilities)

    if have_trace_targets:
        state.tracing.trace_df(
            utilities,
            "%s.utilities" % trace_label,
            column_labels=["alternative", "utility"],
        )

    probs = logit.utils_to_probs(
        state, utilities, trace_label=trace_label, trace_choosers=choosers
    )
    chunk_sizer.log_df(trace_label, "probs", probs)

    del utilities
    chunk_sizer.log_df(trace_label, "utilities", None)

    if have_trace_targets:
        # report these now in case make_choices throws error on bad_choices
        state.tracing.trace_df(
            probs,
            "%s.probs" % trace_label,
            column_labels=["alternative", "probability"],
        )

    if custom_chooser:
        choices, rands = custom_chooser(state, probs, choosers, spec, trace_label)
    else:
        choices, rands = logit.make_choices(state, probs, trace_label=trace_label)

    del probs
    chunk_sizer.log_df(trace_label, "probs", None)

    if have_trace_targets:
        state.tracing.trace_df(
            choices, "%s.choices" % trace_label, columns=[None, trace_choice_name]
        )
        state.tracing.trace_df(rands, "%s.rands" % trace_label, columns=[None, "rand"])

    return choices


def eval_nl(
    state: workflow.State,
    choosers,
    spec,
    nest_spec,
    locals_d,
    custom_chooser: CustomChooser_T,
    estimator,
    log_alt_losers=False,
    want_logsums=False,
    trace_label=None,
    trace_choice_name=None,
    trace_column_names=None,
    *,
    chunk_sizer: chunk.ChunkSizer,
    compute_settings: ComputeSettings | None = None,
):
    """
    Run a nested-logit simulation for when the model spec does not involve alternative
    specific data, e.g. there are no interactions with alternative
    properties and no need to sample from alternatives.

    Parameters
    ----------
    state : workflow.State
        The workflow state object.
    choosers : pandas.DataFrame
        DataFrame of choosers.
    spec : pandas.DataFrame
        A table of variable specifications and coefficient values.
        Variable expressions should be in the table index and the table
        should have a column for each alternative.
    nest_spec : dict
        Dictionary specifying nesting structure and nesting coefficients.
    locals_d : dict or None
        Dictionary of local variables for expression evaluation.
    custom_chooser : function
        Custom alternative to logit.make_choices.
    estimator : Estimator
        Estimator object for estimation mode.
    log_alt_losers : bool
        Write out expressions when all alternatives are unavailable.
    want_logsums : bool
        Whether to return logsums instead of choices.
    trace_label : str or None
        Label for tracing/logging.
    trace_choice_name : str or None
        Column label for trace file csv dump of choices.
    trace_column_names : str or list of str or None
        Chooser columns to include when tracing expression_values.
    chunk_sizer : ChunkSizer
        ChunkSizer object for logging.
    compute_settings : ComputeSettings or None
        Settings for sharrow. If not given, the default settings are used.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Index will be that of `choosers`, values will match the columns of `spec`.
        If want_logsums is True, returns a DataFrame with choices and logsums.
    """

    trace_label = tracing.extend_trace_label(trace_label, "eval_nl")
    assert trace_label
    have_trace_targets = state.tracing.has_trace_targets(choosers)

    logit.validate_nest_spec(nest_spec, trace_label)

    if have_trace_targets:
        state.tracing.trace_df(choosers, "%s.choosers" % trace_label)

    choosers, spec_sh = _preprocess_tvpb_logsums_on_choosers(choosers, spec, locals_d)

    raw_utilities = eval_utilities(
        state,
        spec_sh,
        choosers,
        locals_d,
        log_alt_losers=log_alt_losers,
        trace_label=trace_label,
        have_trace_targets=have_trace_targets,
        estimator=estimator,
        trace_column_names=trace_column_names,
        spec_sh=spec_sh,
        chunk_sizer=chunk_sizer,
        compute_settings=compute_settings,
    )
    chunk_sizer.log_df(trace_label, "raw_utilities", raw_utilities)

    if have_trace_targets:
        state.tracing.trace_df(
            raw_utilities,
            "%s.raw_utilities" % trace_label,
            column_labels=["alternative", "utility"],
        )

    # exponentiated utilities of leaves and nests
    nested_exp_utilities = compute_nested_exp_utilities(raw_utilities, nest_spec)
    chunk_sizer.log_df(trace_label, "nested_exp_utilities", nested_exp_utilities)

    del raw_utilities
    chunk_sizer.log_df(trace_label, "raw_utilities", None)

    if have_trace_targets:
        state.tracing.trace_df(
            nested_exp_utilities,
            "%s.nested_exp_utilities" % trace_label,
            column_labels=["alternative", "utility"],
        )

    # probabilities of alternatives relative to siblings sharing the same nest
    nested_probabilities = compute_nested_probabilities(
        state, nested_exp_utilities, nest_spec, trace_label=trace_label
    )
    chunk_sizer.log_df(trace_label, "nested_probabilities", nested_probabilities)

    if want_logsums:
        # logsum of nest root
        logsums = pd.Series(np.log(nested_exp_utilities.root), index=choosers.index)
        chunk_sizer.log_df(trace_label, "logsums", logsums)

    del nested_exp_utilities
    chunk_sizer.log_df(trace_label, "nested_exp_utilities", None)

    if have_trace_targets:
        state.tracing.trace_df(
            nested_probabilities,
            "%s.nested_probabilities" % trace_label,
            column_labels=["alternative", "probability"],
        )

    # global (flattened) leaf probabilities based on relative nest coefficients (in spec order)
    base_probabilities = compute_base_probabilities(
        nested_probabilities, nest_spec, spec
    )
    chunk_sizer.log_df(trace_label, "base_probabilities", base_probabilities)

    del nested_probabilities
    chunk_sizer.log_df(trace_label, "nested_probabilities", None)

    if have_trace_targets:
        state.tracing.trace_df(
            base_probabilities,
            "%s.base_probabilities" % trace_label,
            column_labels=["alternative", "probability"],
        )

    # note base_probabilities could all be zero since we allowed all probs for nests to be zero
    # check here to print a clear message but make_choices will raise error if probs don't sum to 1
    BAD_PROB_THRESHOLD = 0.001
    no_choices = (base_probabilities.sum(axis=1) - 1).abs() > BAD_PROB_THRESHOLD

    if no_choices.any():
        logit.report_bad_choices(
            state,
            no_choices,
            base_probabilities,
            trace_label=tracing.extend_trace_label(trace_label, "bad_probs"),
            trace_choosers=choosers,
            msg="base_probabilities do not sum to one",
        )

    if custom_chooser:
        choices, rands = custom_chooser(
            state,
            base_probabilities,
            choosers,
            spec,
            trace_label,
        )
    else:
        choices, rands = logit.make_choices(
            state, base_probabilities, trace_label=trace_label
        )

    del base_probabilities
    chunk_sizer.log_df(trace_label, "base_probabilities", None)

    if have_trace_targets:
        state.tracing.trace_df(
            choices, "%s.choices" % trace_label, columns=[None, trace_choice_name]
        )
        state.tracing.trace_df(rands, f"{trace_label}.rands", columns=[None, "rand"])
        if want_logsums:
            state.tracing.trace_df(
                logsums, f"{trace_label}.logsums", columns=[None, "logsum"]
            )

    if want_logsums:
        choices = choices.to_frame("choice")
        choices["logsum"] = logsums

    return choices


@workflow.func
def _simple_simulate(
    state: workflow.State,
    choosers,
    spec,
    nest_spec,
    skims=None,
    locals_d=None,
    custom_chooser: CustomChooser_T = None,
    log_alt_losers=False,
    want_logsums=False,
    estimator=None,
    trace_label=None,
    trace_choice_name=None,
    trace_column_names=None,
    *,
    chunk_sizer,
    compute_settings: ComputeSettings | None = None,
):
    """
    Run an MNL or NL simulation for when the model spec does not involve alternative
    specific data, e.g. there are no interactions with alternative
    properties and no need to sample from alternatives.

    Parameters
    ----------
    state : workflow.State
        The workflow state object.
    choosers : pandas.DataFrame
        DataFrame of choosers.
    spec : pandas.DataFrame
        A table of variable specifications and coefficient values.
        Variable expressions should be in the table index and the table
        should have a column for each alternative.
    nest_spec : dict or None
        For nested logit: dictionary specifying nesting structure and coefficients.
        For multinomial logit: None.
    skims : object, optional
        Skims object for OD impedance matrices.
    locals_d : dict, optional
        Dictionary of local variables for expression evaluation.
    custom_chooser : function, optional
        Custom alternative to logit.make_choices.
    log_alt_losers : bool, optional
        Write out expressions when all alternatives are unavailable.
    want_logsums : bool, optional
        Whether to return logsums instead of choices.
    estimator : function, optional
        Called to report intermediate table results (used for estimation).
    trace_label : str, optional
        Label for tracing/logging.
    trace_choice_name : str, optional
        Column label for trace file csv dump of choices.
    trace_column_names : str or list of str, optional
        Chooser columns to include when tracing expression_values.
    chunk_sizer : ChunkSizer
        ChunkSizer object for logging.
    compute_settings : ComputeSettings or None
        Settings for sharrow. If not given, the default settings are used.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Index will be that of `choosers`, values will match the columns of `spec`.
    """

    if skims is not None:
        set_skim_wrapper_targets(choosers, skims)

    # check if tracing is enabled and if we have trace targets
    have_trace_targets = state.tracing.has_trace_targets(choosers)

    sharrow_enabled = state.settings.sharrow

    if compute_settings is None:
        compute_settings = ComputeSettings()

    # if tracing is not enabled, drop unused columns
    # if not estimation mode, drop unused columns
    if (
        (not have_trace_targets)
        and (estimator is None)
        and (compute_settings.drop_unused_columns)
    ):
        # drop unused variables in chooser table
        choosers = util.drop_unused_columns(
            choosers,
            spec,
            locals_d,
            custom_chooser,
            sharrow_enabled=sharrow_enabled,
            additional_columns=compute_settings.protect_columns,
        )

    if nest_spec is None:
        choices = eval_mnl(
            state,
            choosers,
            spec,
            locals_d,
            custom_chooser,
            log_alt_losers=log_alt_losers,
            want_logsums=want_logsums,
            estimator=estimator,
            trace_label=trace_label,
            trace_choice_name=trace_choice_name,
            trace_column_names=trace_column_names,
            chunk_sizer=chunk_sizer,
            compute_settings=compute_settings,
        )
    else:
        choices = eval_nl(
            state,
            choosers,
            spec,
            nest_spec,
            locals_d,
            custom_chooser,
            log_alt_losers=log_alt_losers,
            want_logsums=want_logsums,
            estimator=estimator,
            trace_label=trace_label,
            trace_choice_name=trace_choice_name,
            trace_column_names=trace_column_names,
            chunk_sizer=chunk_sizer,
            compute_settings=compute_settings,
        )

    return choices


def tvpb_skims(skims):
    """
    Return a list of TransitVirtualPathLogsumWrapper skims from a skims object.

    Parameters
    ----------
    skims : object, list, or dict
        Skims object or collection of skims.

    Returns
    -------
    list
        List of TransitVirtualPathLogsumWrapper objects.
    """

    skims = (
        skims
        if isinstance(skims, list)
        else skims.values()
        if isinstance(skims, dict)
        else [skims]
    )

    # assume any object in skims can be treated as a skim
    for skim in skims:
        try:
            skim.set_df(df)
        except AttributeError:
            pass


def simple_simulate(
    state: workflow.State,
    choosers,
    spec,
    nest_spec,
    skims=None,
    locals_d=None,
    custom_chooser=None,
    log_alt_losers=False,
    want_logsums=False,
    estimator=None,
    trace_label=None,
    trace_choice_name=None,
    trace_column_names=None,
    compute_settings: ComputeSettings | None = None,
):
    """
    Run an MNL or NL simulation for when the model spec does not involve alternative
    specific data, e.g. there are no interactions with alternative
    properties and no need to sample from alternatives.

    Parameters
    ----------
    state : workflow.State
        The workflow state object.
    choosers : pandas.DataFrame
        DataFrame of choosers.
    spec : pandas.DataFrame
        A table of variable specifications and coefficient values.
        Variable expressions should be in the table index and the table
        should have a column for each alternative.
    nest_spec : dict or None
        For nested logit: dictionary specifying nesting structure and coefficients.
        For multinomial logit: None.
    skims : object, optional
        Skims object for OD impedance matrices.
    locals_d : dict, optional
        Dictionary of local variables for expression evaluation.
    custom_chooser : function, optional
        Custom alternative to logit.make_choices.
    log_alt_losers : bool, optional
        Write out expressions when all alternatives are unavailable.
    want_logsums : bool, optional
        Whether to return logsums instead of choices.
    estimator : function, optional
        Called to report intermediate table results (used for estimation).
    trace_label : str, optional
        Label for tracing/logging.
    trace_choice_name : str, optional
        Column label for trace file csv dump of choices.
    trace_column_names : str or list of str, optional
        Chooser columns to include when tracing expression_values.
    compute_settings : ComputeSettings or None
        Settings for sharrow. If not given, the default settings are used.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Index will be that of `choosers`, values will match the columns of `spec`.
    """

    result_list = []
    # segment by person type and pick the right spec for each person type
    for (
        _i,
        chooser_chunk,
        chunk_trace_label,
        chunk_sizer,
    ) in chunk.adaptive_chunked_choosers(state, choosers, trace_label):
        choices = _simple_simulate(
            state,
            chooser_chunk,
            spec,
            nest_spec,
            skims=skims,
            locals_d=locals_d,
            custom_chooser=custom_chooser,
            log_alt_losers=log_alt_losers,
            want_logsums=want_logsums,
            estimator=estimator,
            trace_label=chunk_trace_label,
            trace_choice_name=trace_choice_name,
            trace_column_names=trace_column_names,
            chunk_sizer=chunk_sizer,
            compute_settings=compute_settings,
        )

        result_list.append(choices)

        chunk_sizer.log_df(trace_label, "result_list", result_list)

    if len(result_list) > 1:
        choices = pd.concat(result_list)

    assert len(choices.index == len(choosers.index))

    return choices


def simple_simulate_by_chunk_id(
    state: workflow.State,
    choosers,
    spec,
    nest_spec,
    skims=None,
    locals_d=None,
    custom_chooser=None,
    log_alt_losers=False,
    want_logsums=False,
    estimator=None,
    trace_label=None,
    trace_choice_name=None,
    compute_settings: ComputeSettings | None = None,
):
    """
    Chunk-by-chunk-id wrapper for simple_simulate.

    Parameters
    ----------
    state : workflow.State
        The workflow state object.
    choosers : pandas.DataFrame
        DataFrame of choosers.
    spec : pandas.DataFrame
        A table of variable specifications and coefficient values.
    nest_spec : dict or None
        Nest specification or None.
    skims : object, optional
        Skims object for OD impedance matrices.
    locals_d : dict, optional
        Dictionary of local variables for expression evaluation.
    custom_chooser : function, optional
        Custom alternative to logit.make_choices.
    log_alt_losers : bool, optional
        Write out expressions when all alternatives are unavailable.
    want_logsums : bool, optional
        Whether to return logsums instead of choices.
    estimator : function, optional
        Called to report intermediate table results (used for estimation).
    trace_label : str, optional
        Label for tracing/logging.
    trace_choice_name : str, optional
        Column label for trace file csv dump of choices.
    compute_settings : ComputeSettings or None
        Settings for sharrow. If not given, the default settings are used.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Index will be that of `choosers`, values will match the columns of `spec`.
    """

    choices = None
    result_list = []
    for (
        _i,
        chooser_chunk,
        chunk_trace_label,
        chunk_sizer,
    ) in chunk.adaptive_chunked_choosers_by_chunk_id(state, choosers, trace_label):
        choices = _simple_simulate(
            state,
            chooser_chunk,
            spec,
            nest_spec,
            skims=skims,
            locals_d=locals_d,
            custom_chooser=custom_chooser,
            log_alt_losers=log_alt_losers,
            want_logsums=want_logsums,
            estimator=estimator,
            trace_label=chunk_trace_label,
            trace_choice_name=trace_choice_name,
            chunk_sizer=chunk_sizer,
            compute_settings=compute_settings,
        )

        result_list.append(choices)

        chunk_sizer.log_df(trace_label, "result_list", result_list)

    if len(result_list) > 1:
        choices = pd.concat(result_list)

    return choices


def eval_mnl_logsums(
    state: workflow.State,
    choosers,
    spec,
    locals_d,
    trace_label=None,
    *,
    chunk_sizer,
    compute_settings: ComputeSettings | None = None,
):
    """
    Like eval_nl except return logsums instead of making choices.

    Parameters
    ----------
    state : workflow.State
        The workflow state object.
    choosers : pandas.DataFrame
        DataFrame of choosers.
    spec : pandas.DataFrame
        A table of variable specifications and coefficient values.
    locals_d : dict
        Dictionary of local variables for expression evaluation.
    trace_label : str, optional
        Label for tracing/logging.
    chunk_sizer : ChunkSizer
        ChunkSizer object for logging.
    compute_settings : ComputeSettings or None
        Settings for sharrow. If not given, the default settings are used.

    Returns
    -------
    pandas.Series
        Index will be that of `choosers`, values will be logsum across spec column values.
    """

    # FIXME - untested and not currently used by any models...

    trace_label = tracing.extend_trace_label(trace_label, "eval_mnl_logsums")
    have_trace_targets = state.tracing.has_trace_targets(choosers)

    logger.debug("running eval_mnl_logsums")

    # trace choosers
    if have_trace_targets:
        state.tracing.trace_df(choosers, "%s.choosers" % trace_label)

    utilities = eval_utilities(
        state,
        spec,
        choosers,
        locals_d,
        trace_label,
        have_trace_targets,
        chunk_sizer=chunk_sizer,
        compute_settings=compute_settings,
    )
    chunk_sizer.log_df(trace_label, "utilities", utilities)

    if have_trace_targets:
        state.tracing.trace_df(
            utilities,
            "%s.raw_utilities" % trace_label,
            column_labels=["alternative", "utility"],
        )

    # - logsums
    # logsum is log of exponentiated utilities summed across columns of each chooser row
    logsums = np.log(np.exp(utilities.values).sum(axis=1))
    logsums = pd.Series(logsums, index=choosers.index)
    chunk_sizer.log_df(trace_label, "logsums", logsums)

    # trace utilities
    if have_trace_targets:
        state.tracing.trace_df(
            logsums, "%s.logsums" % trace_label, column_labels=["alternative", "logsum"]
        )

    return logsums


def _preprocess_tvpb_logsums_on_choosers(choosers, spec, locals_d):
    """
    Compute TVPB logsums and attach those values to the choosers.

    Also generate a modified spec that uses the replacement value instead of
    regenerating the logsums dynamically inline.

    Parameters
    ----------
    choosers : pandas.DataFrame
        DataFrame of choosers.
    spec : pandas.DataFrame
        Model spec DataFrame.
    locals_d : dict
        Dictionary of local variables for expression evaluation.

    Returns
    -------
    tuple
        (choosers, spec) with TVPB logsum columns preloaded if needed.
    """
    spec_sh = spec.copy()

    def _replace_in_level(multiindex, level_name, *args, **kwargs):
        y = multiindex.levels[multiindex.names.index(level_name)].str.replace(
            *args, **kwargs
        )
        return multiindex.set_levels(y, level=level_name)

    # Preprocess TVPB logsums outside sharrow
    if "tvpb_logsum_odt" in locals_d:
        tvpb = locals_d["tvpb_logsum_odt"]
        path_types = tvpb.tvpb.network_los.setting(
            f"TVPB_SETTINGS.{tvpb.recipe}.path_types"
        ).keys()
        assignments = {}
        for path_type in ["WTW", "DTW"]:
            if path_type not in path_types:
                continue
            re_spec = spec_sh.index
            re_spec = _replace_in_level(
                re_spec,
                "Expression",
                rf"tvpb_logsum_odt\['{path_type}'\]",
                f"df.PRELOAD_tvpb_logsum_odt_{path_type}",
                regex=True,
            )
            if not all(spec_sh.index == re_spec):
                spec_sh.index = re_spec
                preloaded = locals_d["tvpb_logsum_odt"][path_type]
                assignments[f"PRELOAD_tvpb_logsum_odt_{path_type}"] = preloaded
        if assignments:
            choosers = choosers.assign(**assignments)

    if "tvpb_logsum_dot" in locals_d:
        tvpb = locals_d["tvpb_logsum_dot"]
        path_types = tvpb.tvpb.network_los.setting(
            f"TVPB_SETTINGS.{tvpb.recipe}.path_types"
        ).keys()
        assignments = {}
        for path_type in ["WTW", "WTD"]:
            if path_type not in path_types:
                continue
            re_spec = spec_sh.index
            re_spec = _replace_in_level(
                re_spec,
                "Expression",
                rf"tvpb_logsum_dot\['{path_type}'\]",
                f"df.PRELOAD_tvpb_logsum_dot_{path_type}",
                regex=True,
            )
            if not all(spec_sh.index == re_spec):
                spec_sh.index = re_spec
                preloaded = locals_d["tvpb_logsum_dot"][path_type]
                assignments[f"PRELOAD_tvpb_logsum_dot_{path_type}"] = preloaded
        if assignments:
            choosers = choosers.assign(**assignments)

    return choosers, spec_sh


def eval_nl_logsums(
    state: workflow.State,
    choosers,
    spec,
    nest_spec,
    locals_d,
    trace_label=None,
    *,
    chunk_sizer: chunk.ChunkSizer,
    compute_settings: ComputeSettings | None = None,
):
    """
    Like eval_nl except return logsums instead of making choices.

    Parameters
    ----------
    state : workflow.State
        The workflow state object.
    choosers : pandas.DataFrame
        DataFrame of choosers.
    spec : pandas.DataFrame
        Model spec DataFrame.
    nest_spec : dict
        Nest specification.
    locals_d : dict
        Dictionary of local variables for expression evaluation.
    trace_label : str, optional
        Label for tracing/logging.
    chunk_sizer : ChunkSizer
        ChunkSizer object for logging.
    compute_settings : ComputeSettings or None
        Settings for sharrow. If not given, the default settings are used.

    Returns
    -------
    pandas.Series
        Index will be that of `choosers`, values will be nest logsum based on spec column values.
    """

    trace_label = tracing.extend_trace_label(trace_label, "eval_nl_logsums")
    have_trace_targets = state.tracing.has_trace_targets(choosers)

    logit.validate_nest_spec(nest_spec, trace_label)

    choosers, spec_sh = _preprocess_tvpb_logsums_on_choosers(choosers, spec, locals_d)

    # trace choosers
    if have_trace_targets:
        state.tracing.trace_df(choosers, "%s.choosers" % trace_label)

    raw_utilities = eval_utilities(
        state,
        spec_sh,
        choosers,
        locals_d,
        trace_label=trace_label,
        have_trace_targets=have_trace_targets,
        spec_sh=spec_sh,
        chunk_sizer=chunk_sizer,
        compute_settings=compute_settings,
    )
    chunk_sizer.log_df(trace_label, "raw_utilities", raw_utilities)

    if have_trace_targets:
        state.tracing.trace_df(
            raw_utilities,
            "%s.raw_utilities" % trace_label,
            column_labels=["alternative", "utility"],
        )

    # - exponentiated utilities of leaves and nests
    nested_exp_utilities = compute_nested_exp_utilities(raw_utilities, nest_spec)
    chunk_sizer.log_df(trace_label, "nested_exp_utilities", nested_exp_utilities)

    del raw_utilities  # done with raw_utilities
    chunk_sizer.log_df(trace_label, "raw_utilities", None)

    # - logsums
    logsums = np.log(nested_exp_utilities.root)
    logsums = pd.Series(logsums, index=choosers.index)
    chunk_sizer.log_df(trace_label, "logsums", logsums)

    if have_trace_targets:
        # add logsum to nested_exp_utilities for tracing
        nested_exp_utilities["logsum"] = logsums
        state.tracing.trace_df(
            nested_exp_utilities,
            "%s.nested_exp_utilities" % trace_label,
            column_labels=["alternative", "utility"],
        )
        state.tracing.trace_df(
            logsums, "%s.logsums" % trace_label, column_labels=["alternative", "logsum"]
        )

    del nested_exp_utilities  # done with nested_exp_utilities
    chunk_sizer.log_df(trace_label, "nested_exp_utilities", None)

    return logsums


def _simple_simulate_logsums(
    state: workflow.State,
    choosers,
    spec,
    nest_spec,
    skims=None,
    locals_d=None,
    trace_label=None,
    *,
    chunk_sizer,
    compute_settings: ComputeSettings | None = None,
):
    """
    Like simple_simulate except return logsums instead of making choices.

    Parameters
    ----------
    state : workflow.State
        The workflow state object.
    choosers : pandas.DataFrame
        DataFrame of choosers.
    spec : pandas.DataFrame
        Model spec DataFrame.
    nest_spec : dict or None
        Nest specification or None.
    skims : object, optional
        Skims object for OD impedance matrices.
    locals_d : dict, optional
        Dictionary of local variables for expression evaluation.
    trace_label : str, optional
        Label for tracing/logging.
    chunk_sizer : ChunkSizer
        ChunkSizer object for logging.
    compute_settings : ComputeSettings or None
        Settings for sharrow. If not given, the default settings are used.

    Returns
    -------
    pandas.Series
        Index will be that of `choosers`, values will be nest logsum based on spec column values.
    """

    if skims is not None:
        set_skim_wrapper_targets(choosers, skims)

    # check if tracing is enabled and if we have trace targets
    have_trace_targets = state.tracing.has_trace_targets(choosers)

    if compute_settings is None:
        compute_settings = ComputeSettings()

    # if tracing is not enabled, drop unused columns
    if (not have_trace_targets) and (compute_settings.drop_unused_columns):
        # drop unused variables in chooser table
        choosers = util.drop_unused_columns(
            choosers,
            spec,
            locals_d,
            custom_chooser=None,
            sharrow_enabled=state.settings.sharrow,
            additional_columns=compute_settings.protect_columns,
        )

    if nest_spec is None:
        logsums = eval_mnl_logsums(
            state,
            choosers,
            spec,
            locals_d,
            trace_label=trace_label,
            chunk_sizer=chunk_sizer,
            compute_settings=compute_settings,
        )
    else:
        logsums = eval_nl_logsums(
            state,
            choosers,
            spec,
            nest_spec,
            locals_d,
            trace_label=trace_label,
            chunk_sizer=chunk_sizer,
            compute_settings=compute_settings,
        )

    return logsums


@workflow.func
def simple_simulate_logsums(
    state: workflow.State,
    choosers,
    spec,
    nest_spec,
    skims=None,
    locals_d=None,
    chunk_size=0,
    trace_label=None,
    chunk_tag=None,
    explicit_chunk_size=0,
    compute_settings: ComputeSettings | None = None,
):
    """
    Like simple_simulate except return logsums instead of making choices.

    Parameters
    ----------
    state : workflow.State
        The workflow state object.
    choosers : pandas.DataFrame
        DataFrame of choosers.
    spec : pandas.DataFrame
        Model spec DataFrame.
    nest_spec : dict or None
        Nest specification or None.
    skims : object, optional
        Skims object for OD impedance matrices.
    locals_d : dict, optional
        Dictionary of local variables for expression evaluation.
    chunk_size : int, optional
        Chunk size for adaptive chunking.
    trace_label : str, optional
        Label for tracing/logging.
    chunk_tag : str, optional
        Tag for chunking.
    explicit_chunk_size : int, optional
        Explicit chunk size for adaptive chunking.
    compute_settings : ComputeSettings or None
        Settings for sharrow. If not given, the default settings are used.

    Returns
    -------
    pandas.Series
        Index will be that of `choosers`, values will be nest logsum based on spec column values.
    """

    assert len(choosers) > 0
    chunk_tag = chunk_tag or trace_label

    result_list = []
    # segment by person type and pick the right spec for each person type
    for (
        _i,
        chooser_chunk,
        chunk_trace_label,
        chunk_sizer,
    ) in chunk.adaptive_chunked_choosers(
        state,
        choosers,
        trace_label,
        chunk_tag,
        chunk_size=chunk_size,
        explicit_chunk_size=explicit_chunk_size,
    ):
        logsums = _simple_simulate_logsums(
            state,
            chooser_chunk,
            spec,
            nest_spec,
            skims,
            locals_d,
            chunk_trace_label,
            chunk_sizer=chunk_sizer,
            compute_settings=compute_settings,
        )

        result_list.append(logsums)

        chunk_sizer.log_df(trace_label, "result_list", result_list)

    if len(result_list) > 1:
        logsums = pd.concat(result_list)

    assert len(logsums.index == len(choosers.index))

    return logsums
