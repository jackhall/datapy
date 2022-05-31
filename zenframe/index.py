import copy
from dataclasses import dataclass, replace
import functools as ft
import itertools as it
from multiprocessing.sharedctypes import Value
import operator as op
import typing as ty

from .common import delegate


FROM_IDX = ty.TypeVar('FROM_IDX')
TO_IDX = ty.TypeVar('TO_IDX')


class ComposeableIndex(ty.Collection[FROM_IDX], ty.Protocol[TO_IDX]):
    """ Encapsulates a mapping from user-specified index values to indicies
    to a np.ndarray. This mapping can be right-composed with others to create new indexes.
    The more convoluted the composition, the more expensive it will become to access the
    underlying numpy arrays, until the user decides to reshape.

    This index protocol can't handle slices yet, but eventually it should.
    For this reason, an index cannot be a `ty.Mapping`.
    """
    def __getitem__(self, idx: FROM_IDX) -> TO_IDX:
        """ Returns whatever is needed to index a numpy array. """
        raise NotImplementedError

    def items(self) -> ty.Iterable[ty.Tuple[FROM_IDX, TO_IDX]]:
        return zip(self, map(self.find, self))

    def __eq__(self, other: 'ComposeableIndex') -> bool:
        return all(x == y for x, y in it.zip_longest(self.items(), other.items()))

    def __hash__(self) -> int:
        return ft.reduce(op.xor, map(hash, self.items()))
    
    def mask(self, to_remove: ty.Mapping[FROM_IDX, bool]) -> 'ComposeableIndex':
        raise NotImplementedError


def compatible(left: ComposeableIndex, right: ComposeableIndex) -> bool:
    return all(left[idx] in right for idx in left)


def compose(left: ComposeableIndex, right: ComposeableIndex, verify: bool = False) -> ComposeableIndex:
    """ True if the two indexes can be composed safely. """
    if verify and not compatible(left, right):
        raise IndexError('the domain of inner does not match the codomain of self')

    new_index = copy.copy(left)

    def composed_getitem(self, idx):
        return right[self[idx]]

    new_index.__getitem__ = composed_getitem
    return new_index


def _reraise_index_error(*errors, method=None):
    if method is None:
        return ft.partial(_reraise_index_error, *errors)

    @ft.wraps(method)
    def new_method(self, idx):
        try:
            return method(self, idx)
        except errors as err:
            raise IndexError(idx) from err

    return new_method


def _check_contains_first(find_method):
    @ft.wraps(find_method)
    def new_find_method(self, idx):
        if idx in self:
            return find_method(self, idx)
        else:
            raise IndexError(idx)

    return new_find_method


@dataclass(frozen=True)
class SequenceIndex(ComposeableIndex[int, TO_IDX], ty.Generic[TO_IDX]):
    """ Useful for sorting when right-composed with an existing index (so TO_IDX==int).
    """
    _idx_seq: ty.Sequence[TO_IDX]

    __len__ = delegate('__len__', '_idx_seq')
    __getitem__ = _reraise_index_error(IndexError,
        method=delegate('__getitem__', '_idx_seq'))

    def __contains__(self, obj):
        return isinstance(obj, int) and (0 <= obj < len(self))

    def __iter__(self):
        return iter(range(len(self._idx_seq)))

    def __post_init__(self):
        if len(self._idx_seq) > len(set(self._idx_seq)):
            raise ValueError('sequence elements must be unique')
    
    def mask(self, to_keep: ty.Mapping[int, bool]) -> 'SequenceIndex':
        new_idx_seq = tuple(filter(to_keep.__getitem__, self._idx_seq))
        return replace(self, _idx_seq=new_idx_seq)


@dataclass(frozen=True)
class DictIndex(ComposeableIndex[FROM_IDX, TO_IDX]):
    _mapping: ty.Mapping[TO_IDX, FROM_IDX]

    __contains__ = delegate('__contains__', '_mapping')
    __iter__ = delegate('__iter__', '_mapping')
    __len__ = delegate('__len__', '_mapping')
    __getitem__ = _reraise_index_error(KeyError,
        method=delegate('__getitem__', '_mapping'))


@dataclass(frozen=True)
class FunctionIndex(ComposeableIndex[FROM_IDX, TO_IDX]):
    function: ty.Callable[[FROM_IDX], TO_IDX]
    domain: ty.AbstractSet[FROM_IDX]

    __contains__ = delegate('__contains__', 'domain')
    __iter__ = delegate('__iter__', 'domain')
    __len__ = delegate('__len__', 'domain')
    __getitem__ = _check_contains_first(delegate('__call__', 'function'))


def coerce_idx(i: int, n: int) -> int:
    positive = i if (i >= 0) else (n + i)
    if positive >= n:
        raise IndexError(i)
    else:
        return positive


@dataclass(frozen=True)
class MatrixIndex(ComposeableIndex[ty.Tuple[int, int], int]):
    nrows: int
    ncols: int

    def __contains__(self, obj):
        try:
            row, col = obj
            return (0 <= row < self.nrows) and (0 <= col < self.ncols)
        except TypeError:
            return False

    def __iter__(self):
        return it.product(range(self.nrows), range(self.ncols))

    def __len__(self):
        return self.nrows * self.ncols

    def __getitem__(self, obj):
        row = coerce_idx(obj[0], self.nrows)
        col = coerce_idx(obj[1], self.ncols)
        return (row * self.ncols) + col
