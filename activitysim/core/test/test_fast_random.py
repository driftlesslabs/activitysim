"""Tests for vector_random_standard_normal, uniform, and gumbel."""

from __future__ import annotations

import numba as nb
import numpy as np
import pytest

from activitysim.core.fast_random._fast_random import FastGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_fast_generator():
    return FastGenerator()


def make_state_array(
    fg: FastGenerator, n_agents: int, base_seed: int = 0
) -> np.ndarray:
    """Seed independent rows through NumPy's public state dictionary."""
    return np.array(
        [fg.get_state_array(base_seed + i) for i in range(n_agents)], dtype=np.uint64
    )


_MIXED_DRAW_GOLDENS = {
    "PCG64": {
        "uniform": np.array(
            [
                0.22733602246716966,
                0.31675833970975287,
                0.7973654573327341,
                0.6762546707509746,
            ]
        ),
        "normal": np.array(
            [
                -0.07534330701052097,
                -0.740884652085609,
                -1.3677927017829434,
                0.6488928021930399,
            ]
        ),
        "following": np.array(
            [0.6727560440146213, 0.9418028652699372, 0.248245714629571]
        ),
    },
    "SFC64": {
        "uniform": np.array(
            [
                0.19120274451709907,
                0.30618034325732313,
                0.49135809785873485,
                0.5734727896970208,
            ]
        ),
        "normal": np.array(
            [
                0.6189500858397424,
                0.40745822130897463,
                -1.7965345809319757,
                -1.3488330297420248,
            ]
        ),
        "following": np.array(
            [0.010584846207991938, 0.09323893259291272, 0.7029554329024037]
        ),
    },
}


@pytest.mark.parametrize("bit_generator", ("PCG64", "SFC64"))
def test_mixed_draw_sequence_matches_numpy_and_golden(bit_generator):
    """Freeze distribution output and state consumption for both generators."""
    generator = FastGenerator(bit_gen=bit_generator)
    state = generator.get_state_array(12345)[None, :]
    uniform = generator.vector_random_standard_uniform(state, shape=4)[0]
    normal = generator.vector_random_standard_normal(state, shape=4)[0]
    following = generator.vector_random_standard_uniform(state, shape=3)[0]

    golden = _MIXED_DRAW_GOLDENS[bit_generator]
    np.testing.assert_array_equal(uniform, golden["uniform"])
    np.testing.assert_array_equal(normal, golden["normal"])
    np.testing.assert_array_equal(following, golden["following"])

    # Runtime parity identifies whether a failure comes from ActivitySim's
    # compiled distributions or an intentional upstream stream change.
    numpy_generator = np.random.Generator(getattr(np.random, bit_generator)(seed=12345))
    np.testing.assert_array_equal(uniform, numpy_generator.random(4))
    np.testing.assert_array_equal(normal, numpy_generator.standard_normal(4))
    np.testing.assert_array_equal(following, numpy_generator.random(3))


# ---------------------------------------------------------------------------
# vector_random_standard_uniform
# ---------------------------------------------------------------------------


class TestVectorRandomStandardUniform:
    """Tests for vector_random_standard_uniform."""

    # --- output dtype and basic shape ---

    def test_returns_float64(self):
        fg = make_fast_generator()
        state = make_state_array(fg, 5)
        result = fg.vector_random_standard_uniform(state)
        assert result.dtype == np.float64

    def test_default_shape_all_agents(self):
        n = 7
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        result = fg.vector_random_standard_uniform(state)
        assert result.shape == (n, 1)

    def test_int_shape(self):
        n = 6
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        result = fg.vector_random_standard_uniform(state, shape=4)
        assert result.shape == (n, 4)

    def test_tuple_shape_1d(self):
        n = 5
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        result = fg.vector_random_standard_uniform(state, shape=(3,))
        assert result.shape == (n, 3)

    def test_tuple_shape_2d(self):
        n = 4
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        result = fg.vector_random_standard_uniform(state, shape=(2, 3))
        assert result.shape == (n, 2, 3)

    # --- values in [0, 1) ---

    def test_values_in_unit_interval(self):
        fg = make_fast_generator()
        state = make_state_array(fg, 50)
        result = fg.vector_random_standard_uniform(state, shape=100)
        assert np.all(result >= 0.0)
        assert np.all(result < 1.0)

    # --- selected_positions ---

    def test_selected_positions_shape(self):
        n = 10
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        sel = np.array([0, 2, 5], dtype=np.intp)
        result = fg.vector_random_standard_uniform(
            state, selected_positions=sel, shape=4
        )
        assert result.shape == (3, 4)

    def test_selected_positions_values_in_unit_interval(self):
        n = 10
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        sel = np.array([1, 4, 7], dtype=np.intp)
        result = fg.vector_random_standard_uniform(
            state, selected_positions=sel, shape=20
        )
        assert np.all(result >= 0.0)
        assert np.all(result < 1.0)

    def test_selected_positions_only_selected_rows_mutated(self):
        n = 5
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        state_copy = state.copy()
        sel = np.array([1, 3], dtype=np.intp)
        fg.vector_random_standard_uniform(state, selected_positions=sel, shape=1)
        # Rows NOT in sel must be unchanged.
        for i in range(n):
            if i not in sel:
                np.testing.assert_array_equal(
                    state[i],
                    state_copy[i],
                    err_msg=f"Row {i} should not have been mutated",
                )
        # Rows IN sel must have changed.
        for i in sel:
            assert not np.array_equal(
                state[i], state_copy[i]
            ), f"Row {i} should have been mutated"

    # --- state mutation ---

    def test_state_mutated_in_place(self):
        fg = make_fast_generator()
        state = make_state_array(fg, 4)
        state_before = state.copy()
        fg.vector_random_standard_uniform(state, shape=3)
        assert not np.array_equal(state, state_before)

    # --- reproducibility ---

    def test_reproducibility(self):
        """Identical initial state must produce identical output."""
        fg = make_fast_generator()
        state_a = make_state_array(fg, 8, base_seed=99)
        state_b = state_a.copy()
        out_a = fg.vector_random_standard_uniform(state_a, shape=10)
        out_b = fg.vector_random_standard_uniform(state_b, shape=10)
        np.testing.assert_array_equal(out_a, out_b)

    def test_different_seeds_differ(self):
        fg = make_fast_generator()
        state_a = make_state_array(fg, 5, base_seed=0)
        state_b = make_state_array(fg, 5, base_seed=1000)
        out_a = fg.vector_random_standard_uniform(state_a, shape=20)
        out_b = fg.vector_random_standard_uniform(state_b, shape=20)
        assert not np.array_equal(out_a, out_b)

    # --- statistical sanity (large sample) ---

    def test_mean_close_to_half(self):
        fg = make_fast_generator()
        state = make_state_array(fg, 200)
        result = fg.vector_random_standard_uniform(state, shape=500)
        assert abs(result.mean() - 0.5) < 0.01

    def test_no_exact_ones(self):
        """U[0, 1) must never produce exactly 1.0."""
        fg = make_fast_generator()
        state = make_state_array(fg, 100)
        result = fg.vector_random_standard_uniform(state, shape=1000)
        assert not np.any(result == 1.0)


# ---------------------------------------------------------------------------
# vector_random_standard_normal
# ---------------------------------------------------------------------------


class TestVectorRandomStandardNormal:
    """Tests for vector_random_standard_normal."""

    # --- output dtype and basic shape ---

    def test_returns_float64(self):
        fg = make_fast_generator()
        state = make_state_array(fg, 5)
        result = fg.vector_random_standard_normal(state)
        assert result.dtype == np.float64

    def test_default_shape_all_agents(self):
        n = 7
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        result = fg.vector_random_standard_normal(state)
        assert result.shape == (n, 1)

    def test_int_shape(self):
        n = 6
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        result = fg.vector_random_standard_normal(state, shape=4)
        assert result.shape == (n, 4)

    def test_tuple_shape_1d(self):
        n = 5
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        result = fg.vector_random_standard_normal(state, shape=(3,))
        assert result.shape == (n, 3)

    def test_tuple_shape_2d(self):
        n = 4
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        result = fg.vector_random_standard_normal(state, shape=(2, 3))
        assert result.shape == (n, 2, 3)

    # --- selected_positions ---

    def test_selected_positions_shape(self):
        n = 10
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        sel = np.array([0, 2, 5], dtype=np.intp)
        result = fg.vector_random_standard_normal(
            state, selected_positions=sel, shape=4
        )
        assert result.shape == (3, 4)

    def test_selected_positions_only_selected_rows_mutated(self):
        n = 5
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        state_copy = state.copy()
        sel = np.array([0, 4], dtype=np.intp)
        fg.vector_random_standard_normal(state, selected_positions=sel, shape=1)
        for i in range(n):
            if i not in sel:
                np.testing.assert_array_equal(
                    state[i],
                    state_copy[i],
                    err_msg=f"Row {i} should not have been mutated",
                )
        for i in sel:
            assert not np.array_equal(
                state[i], state_copy[i]
            ), f"Row {i} should have been mutated"

    # --- state mutation ---

    def test_state_mutated_in_place(self):
        fg = make_fast_generator()
        state = make_state_array(fg, 4)
        state_before = state.copy()
        fg.vector_random_standard_normal(state, shape=3)
        assert not np.array_equal(state, state_before)

    # --- reproducibility ---

    def test_reproducibility(self):
        """Identical initial state must produce identical output."""
        fg = make_fast_generator()
        state_a = make_state_array(fg, 8, base_seed=77)
        state_b = state_a.copy()
        out_a = fg.vector_random_standard_normal(state_a, shape=10)
        out_b = fg.vector_random_standard_normal(state_b, shape=10)
        np.testing.assert_array_equal(out_a, out_b)

    def test_different_seeds_differ(self):
        fg = make_fast_generator()
        state_a = make_state_array(fg, 5, base_seed=0)
        state_b = make_state_array(fg, 5, base_seed=1000)
        out_a = fg.vector_random_standard_normal(state_a, shape=20)
        out_b = fg.vector_random_standard_normal(state_b, shape=20)
        assert not np.array_equal(out_a, out_b)

    # --- statistical sanity (large sample) ---

    def test_mean_close_to_zero(self):
        fg = make_fast_generator()
        state = make_state_array(fg, 200)
        result = fg.vector_random_standard_normal(state, shape=500)
        assert abs(result.mean()) < 0.05

    def test_std_close_to_one(self):
        fg = make_fast_generator()
        state = make_state_array(fg, 200)
        result = fg.vector_random_standard_normal(state, shape=500)
        assert abs(result.std() - 1.0) < 0.05

    def test_values_are_finite(self):
        fg = make_fast_generator()
        state = make_state_array(fg, 50)
        result = fg.vector_random_standard_normal(state, shape=200)
        assert np.all(np.isfinite(result))

    def test_distribution_symmetry(self):
        """Mean of absolute values should be close to sqrt(2/pi) ≈ 0.7979."""
        fg = make_fast_generator()
        expected_mean_abs = np.sqrt(2.0 / np.pi)
        state = make_state_array(fg, 200)
        result = fg.vector_random_standard_normal(state, shape=500)
        assert abs(np.abs(result).mean() - expected_mean_abs) < 0.05


# ---------------------------------------------------------------------------
# vector_random_standard_gumbel
# ---------------------------------------------------------------------------


class TestVectorRandomStandardGumbel:
    """Tests for vector_random_standard_gumbel."""

    def test_returns_float64(self):
        fg = make_fast_generator()
        state = make_state_array(fg, 5)
        result = fg.vector_random_standard_gumbel(state)
        assert result.dtype == np.float64

    def test_default_shape_all_agents(self):
        n = 7
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        result = fg.vector_random_standard_gumbel(state)
        assert result.shape == (n, 1)

    def test_selected_positions_shape(self):
        n = 10
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        sel = np.array([0, 2, 5], dtype=np.intp)
        result = fg.vector_random_standard_gumbel(
            state, selected_positions=sel, shape=4
        )
        assert result.shape == (3, 4)

    def test_selected_positions_only_selected_rows_mutated(self):
        n = 5
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        state_copy = state.copy()
        sel = np.array([0, 4], dtype=np.intp)
        fg.vector_random_standard_gumbel(state, selected_positions=sel, shape=1)
        for i in range(n):
            if i not in sel:
                np.testing.assert_array_equal(
                    state[i],
                    state_copy[i],
                    err_msg=f"Row {i} should not have been mutated",
                )
        for i in sel:
            assert not np.array_equal(
                state[i], state_copy[i]
            ), f"Row {i} should have been mutated"

    def test_matches_transformed_uniform(self):
        fg = make_fast_generator()
        state_a = make_state_array(fg, 8, base_seed=99)
        state_b = state_a.copy()

        out_a = fg.vector_random_standard_gumbel(state_a, shape=10)
        out_b = -np.log(-np.log(fg.vector_random_standard_uniform(state_b, shape=10)))

        np.testing.assert_allclose(out_a, out_b)

    def test_values_are_finite(self):
        fg = make_fast_generator()
        state = make_state_array(fg, 50)
        result = fg.vector_random_standard_gumbel(state, shape=200)
        assert np.all(np.isfinite(result))


@pytest.mark.parametrize("bit_generator", ("PCG64", "SFC64"))
@pytest.mark.parametrize("distribution", ("uniform", "normal", "gumbel"))
@pytest.mark.parametrize("shape", (0, (2, 0, 3)))
@pytest.mark.parametrize("selection", (None, (1,), ()))
def test_zero_draw_dimensions_preserve_state(
    bit_generator, distribution, shape, selection
):
    """Explicit output dimensions handle zero draws and empty row selections."""
    generator = FastGenerator(bit_gen=bit_generator)
    state = np.array([generator.get_state_array(i) for i in range(3)])
    before = state.copy()
    selected = None if selection is None else np.asarray(selection, dtype=np.intp)
    result = getattr(generator, f"vector_random_standard_{distribution}")(
        state, selected_positions=selected, shape=shape
    )
    rows = len(state) if selected is None else len(selected)
    trailing = (shape,) if isinstance(shape, int) else shape
    assert result.shape == (rows, *trailing)
    np.testing.assert_array_equal(state, before)


@pytest.mark.parametrize("bit_generator", ("PCG64", "SFC64"))
@pytest.mark.parametrize("seed", (0, 1, 42, 2**32 + 1, 2**64 - 1))
def test_owned_state_long_mixed_sequence_matches_numpy(bit_generator, seed):
    """Exercise rejection sampling and carry propagation, then compare final states."""
    generator = FastGenerator(bit_gen=bit_generator)
    state = generator.get_state_array(seed)[None, :]
    numpy_generator = np.random.Generator(getattr(np.random, bit_generator)(seed))
    for distribution in ("uniform", "normal", "uniform", "normal"):
        actual = getattr(generator, f"vector_random_standard_{distribution}")(
            state, shape=4096
        )[0]
        expected = (
            numpy_generator.random(4096)
            if distribution == "uniform"
            else numpy_generator.standard_normal(4096)
        )
        # libm rounding in the normal tail can differ across platforms.
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    public = numpy_generator.bit_generator.state["state"]
    if bit_generator == "PCG64":
        expected_state = [
            public["state"] & ((1 << 64) - 1),
            public["state"] >> 64,
            public["inc"] & ((1 << 64) - 1),
            public["inc"] >> 64,
        ]
    else:
        expected_state = public["state"]
    np.testing.assert_array_equal(state[0], np.array(expected_state, dtype=np.uint64))


@pytest.mark.parametrize("bit_generator", ("PCG64", "SFC64"))
def test_generation_requires_only_public_numpy_state(monkeypatch, bit_generator):
    """A seeder with no capsule, CFFI, or ctypes interface must still work."""
    original = getattr(np.random, bit_generator)
    reference = np.random.Generator(original(42)).random(20)

    class PublicStateOnly:
        def __init__(self, seed):
            self.state = original(seed).state

        def __getattr__(self, name):
            raise AssertionError(f"private NumPy interface requested: {name}")

    monkeypatch.setattr(np.random, bit_generator, PublicStateOnly)
    generator = FastGenerator(bit_gen=bit_generator)
    state = generator.get_state_array(42)[None, :]
    np.testing.assert_array_equal(
        generator.vector_random_standard_uniform(state, shape=20)[0], reference
    )


@nb.njit
def _draw_raw_words(next_word, state, count):
    """Exercise the transition directly so low bits are not lost to float conversion."""
    result = np.empty(count, dtype=np.uint64)
    for i in range(count):
        result[i] = next_word(state)
    return result


_MAX_WORD = 2**64 - 1


@pytest.mark.parametrize(
    "bit_generator,words",
    [
        ("PCG64", [0, 0, 1, 0]),
        ("PCG64", [_MAX_WORD, _MAX_WORD, _MAX_WORD, _MAX_WORD]),
        ("PCG64", [_MAX_WORD, 0, 1, 0]),
        ("PCG64", [0, 0, 1, 63 << 58]),
        ("SFC64", [0, 0, 0, 1]),
        ("SFC64", [_MAX_WORD, _MAX_WORD, _MAX_WORD, _MAX_WORD]),
        ("SFC64", [_MAX_WORD, 0, 1, _MAX_WORD]),
        ("SFC64", [0, _MAX_WORD, 1 << 63, 0]),
    ],
)
def test_owned_transitions_match_numpy_at_arithmetic_boundaries(bit_generator, words):
    """Cover 64/128-bit carries, counter wrap, and zero/maximal rotation counts."""
    reference = getattr(np.random, bit_generator)(0)
    public = reference.state
    if bit_generator == "PCG64":
        public["state"] = dict(
            state=words[0] + (words[1] << 64), inc=words[2] + (words[3] << 64)
        )
    else:
        public["state"]["state"] = np.array(words, dtype=np.uint64)
    reference.state = public
    generator = FastGenerator(bit_gen=bit_generator)
    actual = _draw_raw_words(
        generator._next_uint64, np.array(words, dtype=np.uint64), 1024
    )
    np.testing.assert_array_equal(actual, reference.random_raw(1024))
