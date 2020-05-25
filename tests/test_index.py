import hypothesis as hyp
import hypothesis.strategies as some
import pytest

import zenframe as zf


@hyp.given(some.integers(min_value=0, max_value=6))
def test_sequence_index(i):
    n = 5
    baseline = range(n)
    index = zf.SequenceIndex(baseline)

    assert (i in index) == (i in baseline)
    assert list(index) == list(baseline)
    assert len(index) == len(baseline)
    if i in baseline:
        assert index.find(i) == i
        if i != 0:
            assert index.find(-i) == (n - i)
    else:
        with pytest.raises(IndexError):
            index.find(i)

        with pytest.raises(IndexError):
            index.find(-i - 1)


keys = some.text(max_size=5)


@hyp.given(some.dictionaries(keys, some.integers()), keys)
def test_dict_index(baseline, bad_key):
    hyp.assume(bad_key not in baseline)

    index = zf.DictIndex(baseline)
    assert list(index) == list(baseline)
    assert len(index) == len(baseline)

    assert bad_key not in index
    with pytest.raises(IndexError):
        index.find(bad_key)

    try:
        good_key = next(iter(baseline))
    except StopIteration:
        return

    assert good_key in index
    assert index.find(good_key) == baseline[good_key]


@hyp.given(some.integers(min_value=0, max_value=6))
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
        assert index.find(i) == baseline_func(i)
    else:
        with pytest.raises(IndexError):
            index.find(i)
