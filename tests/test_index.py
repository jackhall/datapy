import hypothesis as hyp
import hypothesis.strategies as some
import pytest

import zenframe as zf


@hyp.given(some.integers(min_value=-1, max_value=6))
def test_sequence_index(i):
    baseline = range(5)
    index = zf.SequenceIndex(baseline)

    assert (i in index) == (i in baseline)
    assert list(index) == list(baseline)
    assert len(index) == len(baseline)
    if i in baseline:
        assert index.find(i) == i
    else:
        with pytest.raises(IndexError):
            index.find(i)


@hyp.given()
def test_dict_index(x):
    pass
