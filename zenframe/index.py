import copy
import functools as ft
import itertools as it
import operator as op
import typing as ty

import attr

from .common import delegate


FROM_IDX = ty.TypeVar('FROM_IDX')
TO_IDX = ty.TypeVar('TO_IDX')


class ComposeableIndex(ty.Collection[FROM_IDX], ty.Protocol[TO_IDX]):
    """ Encapsulates a mapping from user-specified index values to indicies
    to a np.ndarray: the `find` method. This mapping can be right-composed with
    others to create new indexes. The more convoluted the composition, the
    more expensive it will become to access the underlying numpy arrays, until
    the user decides to reshape.

    This index protocol can't handle slices yet, but eventually it should.
    For this reason, an index cannot be a `ty.Mapping`.
    """
    def find(self, idx: FROM_IDX) -> TO_IDX:
        """ Returns whatever is needed to index a numpy array. """
        raise NotImplementedError

    def fits_around(self, inner: 'ComposeableIndex') -> bool:
        """ True if this index can be composed with `inner` safely. """
        return all(self.find(idx) in inner for idx in self)

    def compose(self, inner: 'ComposeableIndex', verify: bool = False) -> 'ComposeableIndex':
        if verify and not self.fits_around(inner):
            raise IndexError('the domain of inner does not match the codomain of self')

        new_index = copy.copy(self)

        @ft.wraps(new_index.find)
        def composed(idx):
            return inner.find(self.find(idx))

        new_index.find = composed
        return new_index

    def items(self) -> ty.Iterable[ty.Tuple[FROM_IDX, TO_IDX]]:
        return zip(self, map(self.find, self))

    def __eq__(self, other: 'ComposeableIndex') -> bool:
        return all(x == y for x, y in it.zip_longest(self.items(), other.items()))

    def __hash__(self) -> int:
        return ft.reduce(op.xor, map(hash, self.items()))


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


@attr.s(auto_attribs=True, slots=True, frozen=True)
class SequenceIndex(ComposeableIndex[int, TO_IDX], ty.Generic[TO_IDX]):
    """ Useful for sorting when right-composed with an existing index (so TO_IDX==int). """
    idx_seq: ty.Sequence[TO_IDX] = attr.ib()

    __len__ = delegate('__len__', 'idx_seq')
    find = _reraise_index_error(IndexError, find_method=delegate('__getitem__', 'idx_seq'))

    def __contains__(self, obj):
        return isinstance(obj, int) and (0 <= obj < len(self))

    def __iter__(self):
        return iter(range(len(self.idx_seq)))

    @idx_seq.validator
    def check_unique(self, attribute, value):
        if len(value) > len(set(value)):
            raise ValueError('sequence elements must be unique')


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DictIndex(ComposeableIndex[FROM_IDX, TO_IDX]):
    mapping: ty.Mapping[TO_IDX, FROM_IDX]

    __contains__ = delegate('__contains__', 'mapping')
    __iter__ = delegate('__iter__', 'mapping')
    __len__ = delegate('__len__', 'mapping')
    find = _reraise_index_error(KeyError, find_method=delegate('__getitem__', 'mapping'))


@attr.s(auto_attribs=True, slots=True, frozen=True)
class FunctionIndex(ComposeableIndex[FROM_IDX, TO_IDX]):
    func: ty.Callable[[FROM_IDX], TO_IDX]
    domain: ty.Collection[FROM_IDX]

    __contains__ = delegate('__contains__', 'domain')
    __iter__ = delegate('__iter__', 'domain')
    __len__ = delegate('__len__', 'domain')
    find = _check_contains_first(delegate('__call__', 'func'))
