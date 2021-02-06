import itertools as it

import hypothesis as hyp
import hypothesis.strategies as some
import pytest

import zenframe as zf


@hyp.given(some.integers(min_value=0, max_value=5))
def test_sequence_index(i):
    n = 5
    baseline = range(n)
    index = zf.SequenceIndex(baseline)

    assert (i in index) == (i in baseline)
    assert list(index) == list(baseline)
    assert len(index) == len(baseline)
    if i in baseline:
        assert index[i] == i
        if i != 0:
            assert index[-i] == (n - i)
    else:
        with pytest.raises(IndexError):
            index[i]

        with pytest.raises(IndexError):
            index[-i - 1]


keys = some.text(max_size=5)


@hyp.given(some.dictionaries(keys, some.integers()), keys)
def test_dict_index(baseline, bad_key):
    hyp.assume(bad_key not in baseline)

    index = zf.DictIndex(baseline)
    assert list(index) == list(baseline)
    assert len(index) == len(baseline)

    assert bad_key not in index
    with pytest.raises(IndexError):
        index[bad_key]

    try:
        good_key = next(iter(baseline))
    except StopIteration:
        return

    assert good_key in index
    assert index[good_key] == baseline[good_key]


def indices(n):
    return some.integers(min_value=-n, max_value=n - 1)


@hyp.given(indices(6))  # should reject negative indices
def test_function_index(i):
    n = 5
    baseline_domain = range(n)

    def baseline_func(idx):
        return list(reversed(baseline_domain))[idx] - 2

    index = zf.FunctionIndex(baseline_func, baseline_domain)

    assert (i in index) == (i in baseline_domain)
    assert list(index) == list(baseline_domain)
    assert len(index) == len(baseline_domain)
    if i in baseline_domain:
        assert index[i] == baseline_func(i)
    else:
        with pytest.raises(IndexError):
            index[i]


@hyp.given(indices(4), indices(6))
def test_matrix_index(i, j):
    n, m = 3, 5
    index = zf.MatrixIndex(nrows=n, ncols=m)

    assert set(index) == set(it.product(range(n), range(m)))
    assert len(index) == (n * m)
    assert ((i, j) in index) == ((0 <= i < n) and (0 <= j < m))

    if (i in range(-n - 1, n)) and (j in range(-m - 1, m)):
        row = i if (i >= 0) else (n + i)
        col = j if (j >= 0) else (m + j)
        assert index[i, j] == ((row * m) + col)
    else:
        with pytest.raises(IndexError):
            index[i, j]
