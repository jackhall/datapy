import copy
import functools as ft
import operator as op
import typing as ty


FROM_IDX = ty.TypeVar('FROM_IDX')
TO_IDX = ty.TypeVar('TO_IDX')


class ComposeableIndex(ty.Collection[FROM_IDX], ty.Protocol[TO_IDX]):
    """ Encapsulates a mapping from user-specified index values to indicies
    to a np.ndarray: the `find` method. This mapping can be right-composed with
    others to create new indexes. The more convoluted the composition, the
    more expensive it will become to access the underlying numpy arrays, until
    the user decides to reshape.

    This index protocol can't handle slices yet, but eventually it should.
    """
    def find(self, idx: FROM_IDX) -> TO_IDX:
        """ Returns whatever is needed to index a numpy array. """
        pass

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

    def __hash__(self) -> int:
        codomain = map(self.find, self)
        return ft.reduce(op.xor, map(hash, zip(self, codomain)))


T = ty.TypeVar('T')
U = ty.TypeVar('U')


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
