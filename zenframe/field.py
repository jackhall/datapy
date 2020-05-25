import copy
import typing as ty

import attr
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

    def __setitem__(self, idx: FROM_IDX, value: T) -> None:
        """ if idx exists, replace the value; if not, raise an exception """
        pass

    def map(self, func: ty.Callable[[T], ty.Any]) -> 'IndexedNullableField':
        """ apply `func` to every item """
        pass

    def filter(self, pred: ty.Callable[[T], bool]) -> 'IndexedNullableField':
        """ unindex each element for which `pred` is False (in new Series) """
        pass

    def accum(self, binary_func: ty.Callable[[U, T], U]) -> U:
        """ a more generic form of reduce """
        pass

    def resolve(self) -> 'IndexedNullableField':
        """ copy array and flatten index """
        ...


@attr.s(auto_attribs=True)
class NullableField(ty.Generic[T]):
    _array: np.ndarray = attr.ib()
    _null_mask: np.ndarray = attr.ib()  # where mask is False, elements are null

    def __getitem__(self, idx) -> ty.Optional[T]:
        return self._array[idx] if self._null_mask[idx] else None

    def __setitem__(self, idx, value: ty.Optional[T]) -> None:
        """ if idx exists, replace the value; if not, raise an exception """
        if value is None:
            self._null_mask[idx] = False
        else:
            self._array[idx] = value

    def __contains__(self, value: T) -> bool:
        indices = np.where(self._array == value)[0]
        return any(self._null_mask[idx] for idx in indices)

    def map(self, func: ty.Callable[[T], ty.Any]) -> IndexedNullableField:
        """ apply `func` to every item """
        new_array = copy.copy(self._array)
        for idx, value in np.ndenumerate(self._array):
            if self._null_mask[idx]:
                new_array[idx] = func(value)
        return attr.evolve(self, array=new_array)

    def accum(self, binary_func: ty.Callable[[U, T], U], initializer: U) -> U:
        ...


@attr.s(auto_attribs=True)
class Field(ty.Generic[T]):
    """ sequence of T """
    _field: NullableField[T]
    index: ComposeableIndex

    def __getitem__(self, idx) -> ty.Optional[T]:
        return self._field[self.index.find(idx)]

    def __setitem__(self, idx, value: ty.Optional[T]) -> None:
        self._field[self.index.find(idx)] = value

    def __iter__(self) -> ty.Iterator[T]:
        for idx in self.index:
            yield self[idx]

    __contains__ = delegate('__contains__', '_field')
    __len__ = delegate('__len__', 'index')
    map = delegate('map', '_field')  # won't necessarily iterate in the index order
    accum = delegate('accum', '_field')  # won't necessarily iterate in the index order

    def filter(self, pred: ty.Callable[[T], bool]) -> IndexedNullableField:
        """ unindex each element for which `pred` is False (in new Series) """
        ...

    def resolve(self) -> IndexedNullableField:
        ...
