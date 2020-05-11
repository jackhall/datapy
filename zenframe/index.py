import functools as ft
import typing as ty

import attr

from .common import delegate
from .protocols import (
    ComposeableIndex,
    FROM_IDX,
    TO_IDX,
)


def _reraise_index_error(*errors, find_method=None):
    if find_method is None:
        return ft.partial(_reraise_index_error, *errors)

    @ft.wraps(find_method)
    def new_find_method(self, idx):
        try:
            return find_method(self, idx)
        except errors as err:
            raise IndexError(idx) from err

    return new_find_method


def _check_contains_first(find_method):
    @ft.wraps(find_method)
    def new_find_method(self, idx):
        if idx in self:
            return find_method(self, idx)
        else:
            raise IndexError(idx)

    return new_find_method


# could a variation of this be used for sorting?
@attr.s(auto_attribs=True, slots=True, frozen=True)
class SequenceIndex(ComposeableIndex[FROM_IDX, int]):
    _idx_seq: ty.Sequence[FROM_IDX] = attr.ib()

    __contains__ = delegate('__contains__', '_idx_seq')
    __len__ = delegate('__len__', '_idx_seq')
    find = _reraise_index_error(ValueError, find_method=delegate('index', '_idx_seq'))

    def __iter__(self):
        return iter(sorted(self._idx_seq))

    @_idx_seq.validator
    def check_unique(self, attribute, value):
        if len(value) > len(set(value)):
            raise ValueError('sequence elements must be unique')


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DictIndex(ComposeableIndex[FROM_IDX, TO_IDX]):
    _idx_mapping: ty.Dict[TO_IDX, FROM_IDX]

    __contains__ = delegate('__contains__', '_idx_mapping')
    __iter__ = delegate('__iter__', '_idx_mapping')
    __len__ = delegate('__len__', '_idx_mapping')
    find = _reraise_index_error(KeyError, find_method=delegate('__getitem__', '_idx_mapping'))


@attr.s(auto_attribs=True, slots=True, frozen=True)
class FunctionIndex(ComposeableIndex[FROM_IDX, TO_IDX]):
    _func: ty.Callable[[FROM_IDX], TO_IDX]
    _domain: ty.Collection[FROM_IDX]

    __contains__ = delegate('__contains__', '_domain')
    __iter__ = delegate('__iter__', '_domain')
    __len__ = delegate('__len__', '_domain')
    find = _check_contains_first(delegate('__call__', '_func'))
