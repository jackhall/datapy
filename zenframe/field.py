import copy
from dataclasses import dataclass, replace
import typing as ty

import numpy as np

from .common import delegate
from .index import (
    ComposeableIndex,
)


T = ty.TypeVar('T')
U = ty.TypeVar('U')
FROM_IDX = ty.TypeVar('FROM_IDX')


class IndexedNullableField(ty.Collection[T], ty.Protocol[FROM_IDX]):
    index: ComposeableIndex

    def __getitem__(self, idx: FROM_IDX) -> T:
        pass

    def map(self, func: ty.Callable[[T], ty.Any]) -> 'IndexedNullableField':
        """ apply `func` to every non-null item (lazily) """
        pass

    def filter(self, pred: ty.Callable[[T], bool]) -> 'IndexedNullableField':
        """ unindex each element for which `pred` is False (in new Field) """
        pass

    def accum(self, binary_func: ty.Callable[[U, T], U]) -> U:
        """ a more generic form of reduce """
        pass

    def sort(self, key: ty.Callable[[T, T], bool]) -> 'IndexedNullableField':
        """ right-compose a sequence index """
        pass

    def group_by(self, func: ty.Callable[[FROM_IDX, T], U]) -> ty.Dict[U, 'IndexedNullableField']:
        """ return fields as chunks """
        pass

    def update(self, other: 'IndexedNullableField') -> 'IndexedNullableField':
        """ replace values in this array with values from `other`, where the indices overlap 
        """
        pass

    def resolve(self) -> 'IndexedNullableField':
        """ copy array, flatten index and apply map """
        pass


@dataclass(frozen=True)
class NumpyField(ty.Generic[T]):
    """ sequence of T """
    _array: np.ma.MaskedArray
    index: ComposeableIndex
    fill: ty.Any = None
    _map: ty.Optional[ty.Callable[[ty.Any], T]] = None

    def __getitem__(self, idx) -> ty.Optional[T]:
        # what about slicing?
        raw = self._array[self.index[idx]]
        if raw is np.ma.masked:
            return self.fill
        elif self._map is None:
            return raw
        else:
            return self._map(raw)

    def __iter__(self) -> ty.Iterator[T]:
        return iter(self[idx] for idx in self.index)

    __contains__ = delegate('__contains__', '_array')
    __len__ = delegate('__len__', 'index')

    def map(self, func: ty.Callable[[T], ty.Any], **kwargs) -> IndexedNullableField:
        new_func = func if self._map is None else lambda x: func(self._map(x))
        new_fill = kwargs.get('fill', func(self.fill))
        return replace(self, _map=new_func, fill=new_fill)

    def accum(self, func: ty.Callable[[T, U], U], initialize: U) -> U:
        current = initialize
        for elem in self._array:
            current = func(elem, current)
        return current

    def filter(self, pred: ty.Callable[[T], bool]) -> IndexedNullableField:
        ...

    def sort(self, key: ty.Callable[[T, T], bool]) -> 'IndexedNullableField':
        ...

    def group_by(self, func: ty.Callable[[FROM_IDX, T], U]) -> ty.Dict[U, 'IndexedNullableField']:
        """ return fields as chunks """
        ...

    def update(self, other: 'IndexedNullableField') -> 'IndexedNullableField':
        ...

    def resolve(self) -> IndexedNullableField:
        if self._map is None:
            ...
        else:
            new_array = np.vectorize(self._map)(self._array)
