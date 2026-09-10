"""Import isolation for optional accelerated RNG implementations."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_legacy_rng_does_not_import_accelerated_modules():
    """Legacy channels must work even if loading the accelerated module fails."""
    script = """
        import importlib.abc
        import sys
        import pandas as pd

        class BlockFastRandom(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith("activitysim.core.fast_random"):
                    raise ImportError("accelerated RNG deliberately unavailable")

        sys.meta_path.insert(0, BlockFastRandom())
        from activitysim.core.random import Random
        rows = pd.DataFrame(index=pd.Index([1, 2], name="person_id"))
        rng = Random()
        assert rng.channel_type == "legacy"
        rng.add_channel("persons", rows)
        rng.begin_step("legacy_only")
        assert rng.random_for_df(rows).shape == (2, 1)
        assert not any(name.startswith("activitysim.core.fast_random") for name in sys.modules)
        try:
            Random("pcg64").add_channel("persons", rows)
        except ImportError as error:
            assert "deliberately unavailable" in str(error)
        else:
            raise AssertionError("accelerated channels must load their implementation")
    """
    subprocess.run([sys.executable, "-c", textwrap.dedent(script)], check=True)


def test_importing_accelerated_package_does_not_construct_generators():
    """Importing helpers must not eagerly initialize either entropy generator."""
    script = """
        import numpy as np
        import numba

        def unavailable(*args, **kwargs):
            raise AssertionError("unexpected bit-generator initialization during import")

        np.random.PCG64 = np.random.SFC64 = unavailable
        import activitysim.core.fast_random as package
        from activitysim.core.fast_random import _entropy
        assert package.__all__ == ("FastChannel",)
        assert _entropy._entropy_generator.cache_info().currsize == 0
    """
    subprocess.run([sys.executable, "-c", textwrap.dedent(script)], check=True)
