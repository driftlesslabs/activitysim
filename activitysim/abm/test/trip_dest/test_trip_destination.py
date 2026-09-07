from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from activitysim import abm  # noqa: F401
from activitysim.core import workflow as wf


def run_trip_destination(
    tmp_path: Path,
    explicit_chunk: float | None = None,
    repeat_work_tours: int = 1,
):
    shutil.copytree(
        Path(__file__).parent.joinpath("configs"), tmp_path.joinpath("configs")
    )
    shutil.copytree(Path(__file__).parent.joinpath("data"), tmp_path.joinpath("data"))

    if explicit_chunk is not None:
        with (tmp_path / "configs" / "trip_destination.yaml").open("a") as stream:
            stream.write(f"\nexplicit_chunk: {explicit_chunk}\n")

    state = wf.State.make_default(working_dir=tmp_path)

    # init tours
    tours = pd.read_csv(
        tmp_path / state.filesystem.data_dir[0] / "tours.csv"
    ).set_index("tour_id")
    base_tour = tours.loc[[500]]
    tours = pd.concat(
        [tours]
        + [base_tour.rename(index={500: 510 + i}) for i in range(1, repeat_work_tours)]
    ).sort_index()
    state.add_table("tours", tours)
    state.tracing.register_traceable_table("tours", tours)
    state.get_rn_generator().add_channel("tours", tours)

    # init trips
    trips = pd.read_csv(
        tmp_path / state.filesystem.data_dir[0] / "trips.csv"
    ).set_index("trip_id")
    base_trips = trips[trips.tour_id == 500]
    repeated_trips = [trips]
    for i in range(1, repeat_work_tours):
        trip_copy = base_trips.copy()
        trip_copy.index += 100_000 * i
        trip_copy["tour_id"] = 510 + i
        repeated_trips.append(trip_copy)
    trips = pd.concat(repeated_trips).sort_index()
    state.add_table("trips", trips)
    state.tracing.register_traceable_table("trips", trips)
    state.get_rn_generator().add_channel("trips", trips)

    state.run.all()

    return state.get_dataframe("trips")


def test_trip_destination(tmp_path: Path):
    out_trips = run_trip_destination(tmp_path)

    # logsums are generated for intermediate trips only
    assert out_trips["destination_logsum"].isna().tolist() == [
        True,
        False,
        True,
        True,
        False,
        True,
    ]


def test_trip_destination_chunked_logsums_match_unchunked(tmp_path: Path):
    # Repeat one tour so each work-purpose presample has enough chooser rows to
    # exercise the outer TAZ-to-MAZ pipeline chunker.
    unchunked = run_trip_destination(tmp_path / "unchunked", repeat_work_tours=4)
    chunked = run_trip_destination(
        tmp_path / "chunked",
        explicit_chunk=0.5,
        repeat_work_tours=4,
    )

    pd.testing.assert_frame_equal(chunked, unchunked)
