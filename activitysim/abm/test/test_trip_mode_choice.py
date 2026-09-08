from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from activitysim.abm.models import trip_mode_choice as trip_mode_choice_module


class _DummySkimDict:
    def wrap_3d(self, **_kwargs):
        return object()

    def wrap(self, *_args):
        return object()

    def map_time_periods_from_series(self, periods):
        return periods.map({"AM": 0, "PM": 1})


class _DummyState:
    current_model_name = "test_trip_mode_choice"

    def __init__(self, trips):
        self.settings = SimpleNamespace(
            downcast_int=False,
            downcast_float=False,
            skip_failed_choices=False,
            trace_hh_id=None,
        )
        self.filesystem = SimpleNamespace(
            read_model_spec=lambda **_kwargs: pd.DataFrame(),
            get_segment_coefficients=lambda *_args: {},
        )
        self.tables = {"trips": trips}

    def add_table(self, name, df):
        self.tables[name] = df

    def get_dataframe(self, name, columns=None, as_copy=True):
        df = self.tables[name]
        if columns is not None:
            df = df[columns]
        return df.copy() if as_copy else df

    def is_table(self, _name):
        return False


def test_post_choice_annotations_receive_full_trip_period_then_remove_it(monkeypatch):
    trips = pd.DataFrame(
        {
            "tour_id": [11, 12, 13],
            "household_id": [1, 2, 3],
            "primary_purpose": ["work", "shopping", "work"],
            "depart": [8, 17, 9],
            "origin": [1, 2, 3],
            "destination": [2, 3, 1],
        },
        index=pd.Index([101, 102, 103], name="trip_id"),
    )
    state = _DummyState(trips)
    skim_dict = _DummySkimDict()
    network_los = SimpleNamespace(
        skim_time_periods=SimpleNamespace(period_minutes=60),
        skim_time_period_label=lambda depart, as_cat=True: depart.map(
            lambda value: "AM" if value < 12 else "PM"
        ),
        get_default_skim_dict=lambda: skim_dict,
    )
    model_settings = SimpleNamespace(
        MODE_CHOICE_LOGSUM_COLUMN_NAME="mode_choice_logsum",
        TOURS_MERGED_CHOOSER_COLUMNS=[],
        CHOOSER_COLS_TO_KEEP=[],
        FORCE_ESCORTEE_CHAUFFEUR_MODE_MATCH=False,
        SPEC="trip_mode_choice.csv",
        explicit_chunk=None,
        compute_settings=None,
    )

    monkeypatch.setattr(
        trip_mode_choice_module.tracing, "print_summary", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        trip_mode_choice_module.config, "get_model_constants", lambda *_args: {}
    )
    monkeypatch.setattr(
        trip_mode_choice_module.config, "get_logit_model_settings", lambda *_args: None
    )
    monkeypatch.setattr(
        trip_mode_choice_module.simulate,
        "eval_coefficients",
        lambda _state, spec, *_args: spec,
    )
    monkeypatch.setattr(
        trip_mode_choice_module.simulate,
        "eval_nest_coefficients",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        trip_mode_choice_module.expressions,
        "annotate_preprocessors",
        lambda *_args, **_kwargs: None,
    )

    def choose_mode(_state, choosers, **_kwargs):
        return pd.DataFrame(
            {
                "trip_mode": "DRIVE",
                "mode_choice_logsum": 1.0,
            },
            index=choosers.index,
        )

    monkeypatch.setattr(trip_mode_choice_module, "mode_choice_simulate", choose_mode)

    def annotate_tables(_state, **_kwargs):
        annotated = _state.get_dataframe("trips")
        assert annotated.index.equals(trips.index)
        assert annotated["trip_period"].tolist() == [0, 1, 0]
        annotated["post_choice_skim_value"] = [10.0, 20.0, 30.0]
        _state.add_table("trips", annotated)

    monkeypatch.setattr(
        trip_mode_choice_module.expressions, "annotate_tables", annotate_tables
    )

    trip_mode_choice_module.trip_mode_choice(
        state,
        trips,
        network_los,
        model_settings=model_settings,
    )

    result = state.get_dataframe("trips", as_copy=False)
    assert "trip_period" not in trips
    assert "trip_period" not in result
    assert result["post_choice_skim_value"].tolist() == [10.0, 20.0, 30.0]
