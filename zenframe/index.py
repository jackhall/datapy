import typing as ty

import attr

from .common import delegate
from .protocols import (
    ComposeableIndex,
    FROM_IDX,
    TO_IDX,
)


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DictIndex(ComposeableIndex[FROM_IDX, TO_IDX]):
    _idx_mapping: ty.Dict[TO_IDX, FROM_IDX]

    __contains__ = delegate('__contains__', '_idx_mapping')
    __iter__ = delegate('__iter__', '_idx_mapping')
    __len__ = delegate('__len__', '_idx_mapping')

    def find(self, idx):
        try:
            return self._idx_mapping[idx]
        except KeyError as err:
            raise IndexError(idx) from err


@attr.s(auto_attribs=True, slots=True, frozen=True)
class FunctionIndex(ComposeableIndex[FROM_IDX, TO_IDX]):
    _func: ty.Callable[[FROM_IDX], TO_IDX]
    _domain: ty.Collection[FROM_IDX]

    __contains__ = delegate('__contains__', '_domain')
    __iter__ = delegate('__iter__', '_domain')
    __len__ = delegate('__len__', '_domain')

    def find(self, idx):
        if idx in self._domain:
            return self._func(idx)
        else:
            raise IndexError(idx)
