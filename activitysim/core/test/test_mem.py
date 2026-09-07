from __future__ import annotations

from activitysim.core import mem


def test_release_memory_is_advisory():
    assert isinstance(mem.release_memory(), bool)
