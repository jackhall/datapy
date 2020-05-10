import copy
import typing as ty

import attr
import numpy as np

from .common import delegate
from .protocols import (
    ComposeableIndex,
    IndexedNullableField,
)


T = ty.TypeVar('T')
U = ty.TypeVar('U')


@attr.s(auto_attribs=True)
class Field(ty.Generic[T]):
    """ sequence of T """
    _array: np.ndarray = attr.ib()
    _null_mask: np.ndarray = attr.ib()  # each element is False if None
    index: ComposeableIndex

    def __getitem__(self, idx) -> ty.Optional[T]:
        np_idx = self.index.find(idx)
        if self._null_mask[np_idx]:
            return self._array[np_idx]
        else:
            return None

    def __setitem__(self, idx, value: ty.Optional[T]):
        """ if idx exists, replace the value; if not, raise an exception """
        np_idx = self.index.find(idx)
        if value is None:
            self._null_mask[np_idx] = False
        else:
            self._array[np_idx] = value

    def __iter__(self):
        for idx in self.index:
            yield self[idx]

    def __contains__(self, value: T):
        indices = np.where(self._array == value)[0]
        return any(self._null_mask[idx] for idx in indices)

    __len__ = delegate('__len__', 'index')

    def map(self, func: ty.Callable[[T], ty.Any]) -> IndexedNullableField:
        """ apply `func` to every item """
        new_array = copy.copy(self._array)
        for np_idx, value in np.ndenumerate(self._array):
            if self._null_mask[np_idx]:
                new_array[np_idx] = func(value)
        return attr.evolve(self, array=new_array)

    def filter(self, pred: ty.Callable[[T], bool]) -> IndexedNullableField:
        """ unindex each element for which `pred` is False (in new Series) """
        ...

    def accum(self, binary_func: ty.Callable[[U, T], U]) -> U:
        ...

    def resolve(self) -> IndexedNullableField:
        ...
