import typing as ty

import attr
import numpy as np
import pandas as pd


T = ty.TypeVar('T')


@attr.s(auto_attribs=True)
class ShapeMixin:
    _index: pd.MultiIndex

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        # outer -> inner from index, then a last digit for the number of cols
        # does this make sense for all indexes? no, only dense ones
        ...

    @property
    def is_sparse(self) -> bool:
        ...

    def index(self, idx):
        # return whatever is needed to pull data out of ndarrays
        ...


@attr.s(auto_attribs=True)
class Field(ty.Generic[T], ShapeMixin):
    """ sequence of T """
    _array: np.ndarray = attr.ib()
    _null_mask: np.ndarray = attr.ib()

    def __getitem__(self, idx) -> ty.Optional[T]:
        ...

    def __setitem__(self, idx, value: ty.Optional[T]):
        # if idx exists, replace the value; if not, raise an exception
        ...

    def __contains__(self, value):
        ...

    def __len__(self):
        return len(self._array)

    def T(self, *dims) -> 'Field':  # T := transpose
        ...

    def map(self, func: ty.Callable[T, ty.Any]) -> 'Field':
        """ apply `func` to every item
        copy `_index` to decouple from owning frame?
        """
        ...

    def filter(self, pred: ty.Callable[T, bool]) -> 'Field':
        """ unindex each element for which `pred` is False (in new Series) """
        ...


@attr.s(auto_attribs=True)
class DataFrame(ShapeMixin):
    """
    Rows are hierarchically indexed to provide arbitrary dimensionality.
    Each 'row' is tuple-like.
    Each 'column' is a nullable field, with every element the same type.
    Types:
    - boolean
    - categorical (enum)
    - integer
    - float
    - string
    - datetime
    - object (try not to use this)

    Encourage subclassing to add chained methods.
    """
    fields: ty.Dict[str, Field]  # should I subclass dict? keep it immutable?

    @property
    def f(self):
        return self.fields

    @property
    def r(self):
        return self.rows

    def T(self, *dims) -> 'DataFrame':  # T := transpose
        ...

    @property
    def rows(self) -> ty.Sequence[ty.NamedTuple]:
        ...

    def join(self, other, how, left_on, right_on, suffixes=('_x', '_y'),
             validate=None) -> 'DataFrame':
        ...

    def concat(self, other) -> 'DataFrame':
        ...

    def assign(self, field_name, func) -> 'DataFrame':
        """ like r.map, but returns a new dataframe with one more column """
        ...

    @classmethod
    def from_arrays(cls, arrays: ty.Mapping[str, ty.Sequence],
                    index: pd.MultiIndex, dtypes) -> 'DataFrame':
        ...


R = ty.NamedTuple  # R := row type


@attr.s(auto_attrib=True, slots=True, frozen=True)
class Rows(ty.Generic[R]):
    """ sequence of namedtuples """
    _df: DataFrame

    def map(self, func: ty.Callable[R, ty.Any]) -> Field:
        """ apply func to every row """
        ...

    def filter(self, pred: ty.Callable[R, bool]) -> DataFrame:
        """ unindex each row for which `pred` is False (in new df) """
        ...

    def sort(self, key):
        ...

    def __getitem__(self, idx) -> R:
        ...

    def __setitem__(self, idx, value: R):
        # allow new idx
        ...

    def __contains__(self, idx):
        ...

    def __len__(self):
        ...
